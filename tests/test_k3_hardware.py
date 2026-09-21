"""Opt-in tests: LAYA_K3_TESTS=1, prepared LAYA_K3_HOME, real K3 only."""
import gc
import json
import os
from pathlib import Path
import platform

import pytest

pytestmark = [pytest.mark.k3, pytest.mark.skipif(
    os.environ.get('LAYA_K3_TESTS') != '1' or platform.machine() != 'riscv64',
    reason='requires explicit opt-in on K3 with prepared models')]
FIXTURES = Path(__file__).parent/'fixtures'


@pytest.fixture(autouse=True)
def collect_old_sessions():
    gc.collect()
    yield
    gc.collect()


def workers():
    return {int(path.name) for path in Path('/proc/self/task').iterdir()
            if os.sched_getaffinity(int(path.name)) <= set(range(8,16))}


@pytest.mark.parametrize('full_metrics',[False,True])
def test_sixty_real_moves_match_original_sdk_reference(full_metrics):
    from snake_k3.policy import LayaPolicy
    from snake_k3.game import SnakeGame
    policy = LayaPolicy(backend='spacemit',full_metrics=full_metrics,threads=4)
    reference = [json.loads(s) for s in (FIXTURES/'reference-60.jsonl').read_text().splitlines()]
    game = SnakeGame(12,8,seed=7,initial_length=6)
    times = []
    for frame in reference:
        assert game.snapshot() == frame['game']
        result = policy.decide(game)
        want = frame['decision']
        assert result.proposed == want['proposed']
        assert result.executed == want['executed']
        assert result.probabilities == pytest.approx(want['probabilities'],abs=.002)
        if full_metrics:
            assert result.dead_end_risk == pytest.approx(want['dead_end_risk'],abs=.002)
            assert result.food_reachable == pytest.approx(want['food_reachable'],abs=.002)
        times.append(result.inference_ms)
        game.step(result.executed)
        assert game.alive
    assert game.score == 3
    mean = sum(times[1:])/len(times[1:])
    if not full_metrics:
        assert mean < 140, mean
    print('K3_60',{'full_metrics':full_metrics,'steady_mean_ms':mean})


def test_all_121_inputs_preserve_precision_and_one_session():
    import numpy as np
    import torch
    from snake_k3.policy import LayaPolicy
    policy = LayaPolicy(backend='spacemit',full_metrics=False,threads=4)
    owned = workers()
    assert len(owned) == 4
    manifest = json.loads((FIXTURES/'manifest.json').read_text())
    references = {r['file']:r for r in json.loads((FIXTURES/'baseline-all.json').read_text())['rows'] if r['repeat']==0}
    for item in manifest:
        feed = dict(np.load(FIXTURES/item['file']))
        names = ('input_ids','attention_mask','marker_pos','marker_mask','qtype')
        logits,_ = policy.agent.model(*(torch.from_numpy(feed[k]) for k in names))
        for row,k,want in zip(logits,item['markers'],references[item['file']]['probs']):
            got = row[:k].float().softmax(-1).numpy()
            assert np.max(np.abs(got-want)) < .002, item['file']
        assert workers() == owned, 'Model session/worker pool was replaced'
