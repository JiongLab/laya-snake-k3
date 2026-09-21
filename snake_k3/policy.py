"""Real Laya predictions, with an explicit optional deterministic safety shield."""

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from .game import DIRECTIONS

LAYA_ROOT = Path(os.environ.get('LAYA_K3_HOME',
    str(Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share'))) / 'laya-snake-k3'))).expanduser()
DEFAULT_MODEL = str(LAYA_ROOT / "models" / "multilingual")


def local_checkpoint(value=None):
    """Resolve a directory or an already cached Hub snapshot without network access."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if value is None:
        value = DEFAULT_MODEL
    path = Path(value).expanduser()
    if path.is_dir():
        return path
    if str(value).startswith((".", "/", "~")):
        raise FileNotFoundError(f"Local checkpoint does not exist: {value}")
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(str(value), local_files_only=True))
    except Exception as error:
        raise FileNotFoundError(
            f"{value} is not cached. Download it before starting the offline demo:\n"
            f"  hf download {value} --local-dir models/snake\n"
            "  laya-snake --model models/snake"
        ) from error


def hardware_name():
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip().removeprefix("Apple ")
    return platform.machine()


def checkpoint_metadata(path):
    meta = path / "mlx_config.json"
    values = json.loads(meta.read_text()) if meta.exists() else {}
    manifest = path / "manifest.json"
    source = json.loads(manifest.read_text()) if manifest.exists() else {}
    return {
        "name": values.get("repository", path.name),
        "source_revision": values.get("source_revision"),
        "weight_sha256_from_manifest": source.get("files", {})
        .get("model.safetensors", {})
        .get("sha256"),
        "hardware": hardware_name(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("torch", "laya", "transformers", "numpy", "rich", "tokenizers", "huggingface-hub")
        },
        "engine": "PyTorch CPU FP32",
        "network": "offline",
        "policy": "Laya probabilities over planner features; optional cycle safety shield",
        "source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
    }


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    dead_end_risk: float | None
    food_reachable: float | None
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
    safe_count: int
    planner_best: str

    def to_dict(self):
        return asdict(self)


class LayaPolicy:
    def __init__(self, model=None, *, guarded=True, prompt="compact", optimize=False,
                 threads=4, full_metrics=True, backend='torch'):
        if backend not in ('torch', 'spacemit'):
            raise ValueError('backend must be torch or spacemit')
        if optimize:
            raise ValueError("--optimize is an MLX-only option; this K3 port uses PyTorch eager CPU")
        os.environ["USE_TF"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from .k3_compat import prepare
        prepare()
        import torch
        import laya
        from transformers.initialization import no_init_weights
        from threadpoolctl import threadpool_limits
        if threads not in (1, 2, 4, 8):
            raise ValueError("threads must be 1, 2, 4 or 8")
        torch.set_num_threads(threads)
        self._thread_limits = threadpool_limits(limits=threads)
        self.path = local_checkpoint(model)
        # Every persistent parameter is replaced by the checkpoint. Keep Laya's
        # strict shape/key checks and FP32 copy, but do not generate throwaway
        # random weights first. Nonpersistent RoPE buffers are built normally.
        backend_metadata = {}
        if backend == 'spacemit':
            if self.path.resolve() != Path(DEFAULT_MODEL).resolve():
                raise ValueError('AI Core currently supports the installed multilingual checkpoint only')
            from .onnx_backend import load_spacemit_agent
            self.agent, backend_metadata = load_spacemit_agent(self.path, threads)
        else:
            with no_init_weights():
                self.agent = laya.load(str(self.path), device="cpu")
        self.guarded = guarded
        self.full_metrics = full_metrics
        if prompt not in ("compact", "detailed"):
            raise ValueError("prompt must be compact or detailed")
        self.prompt = prompt
        self.metadata = checkpoint_metadata(self.path)
        self.metadata["prompt"] = prompt
        self.metadata["optimization"] = f"eager CPU, {threads} threads, skip random initialization"
        self.metadata["questions_per_move"] = 3 if full_metrics else 1
        self.metadata.update(backend_metadata)

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe and self.guarded:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, space = game.food_reachability()
        descriptions = {}
        for move in moves:
            if not move.legal:
                descriptions[move.direction] = f"Collision: {move.reason}. Unsafe."
            elif not move.safe:
                descriptions[move.direction] = "Unsafe route. Risk of trapping the snake."
            elif move.eats:
                descriptions[move.direction] = "Safe. Eat the food immediately. Best move."
            elif move.direction == preferred:
                descriptions[move.direction] = "Safe. Best progress toward food."
            else:
                descriptions[move.direction] = "Safe but less progress toward food."
        state = (
            f"Snake game. {len(safe)} safe directions available. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}. "
            f"Open cells: {space}. Snake length: {len(game.body)}. "
            f"{'There is a safe route forward.' if safe else 'The snake is trapped.'}"
        )
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Select the safest move with best progress toward food. Avoid collisions.",
                "criteria": descriptions,
            },
            "risk": {
                "type": "noul",
                "instructions": "Is there a safe route forward for the snake?",
            },
            "food": {
                "type": "noul",
                "instructions": "Is food reachable through the currently empty cells?",
            },
        }
        if self.prompt == "compact":
            state = (
                f"Safe route: {'yes' if safe else 'no'}. "
                f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
            )
            questions["move"]["instructions"] = "Choose the best safe move toward food."
            questions["move"]["criteria"] = {
                m.direction: (
                    "Blocked. Collision."
                    if not m.legal
                    else "Unsafe. Traps the snake."
                    if not m.safe
                    else "Safe. Eat food now. Best."
                    if m.eats
                    else "Safe. Best route to food."
                    if m.direction == preferred
                    else "Safe. Slower route."
                )
                for m in moves
            }
            questions["risk"]["instructions"] = "Is a safe route available?"
            questions["food"]["instructions"] = "Is food reachable through empty cells?"
        inference_start = time.perf_counter()
        selected = questions if self.full_metrics else {"move": questions["move"]}
        output = self.agent.predict(state, selected)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        answers = output["answers"]
        probabilities = answers["move"]["probabilities"]
        scores = list(probabilities.values())
        if self.full_metrics:
            scores.extend((answers["risk"]["noul"], answers["food"]["noul"]))
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Model returned an invalid probability; no move executed")
        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and proposed not in allowed
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=1 - answers["risk"]["noul"] if self.full_metrics else None,
            food_reachable=answers["food"]["noul"] if self.full_metrics else None,
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=output["usage"]["input_tokens"],
            output_tokens=output["usage"].get("output_tokens", 0),
            safe_count=len(safe),
            planner_best=preferred,
        )
