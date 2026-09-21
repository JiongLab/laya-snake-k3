"""K3 AI-core matrix offload with the original Laya SDK input/output contract."""
import json
from pathlib import Path
import sys

# One fast graph for all tested lengths/batches. EP Transpose caused batch
# probability errors; keep it on CPU without disabling the other fast kernels.
# Never rebuild this session in response to an input shape change.
CPU_OPS = 'Transpose;Concat;Div;Erf;MultiHeadMatMul;Split;Gather;Relu;Cos;Sin;Where;IsNaN'


def load_spacemit_agent(model_dir, threads):
    import torch
    from transformers import AutoTokenizer
    from laya.agent import Agent

    # Debian installs the vendor build separately from its generic ORT package.
    # Select it for this process only; never replace the system's generic build.
    vendor = Path(f'/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages')
    if not (vendor / 'spacemit_ort').is_dir():
        raise RuntimeError('Install python3-spacemit-ort and spacemit-onnxruntime to use AI Core')
    sys.path.insert(0, str(vendor))
    import onnxruntime as ort
    if '+spacemit' not in ort.__version__:
        raise RuntimeError('Generic ONNX Runtime is already loaded. Start a fresh process for AI Core.')
    import spacemit_ort

    path = Path(model_dir).parent / 'multilingual-onnx-comparison' / 'model.onnx'
    if not path.is_file():
        raise FileNotFoundError(path)
    affinity = '8;10;12;14' if threads == 4 else ';'.join(str(i) for i in range(8,8+threads))
    ep_options = {'SPACEMIT_EP_INTRA_THREAD_NUM': str(threads),
                  'SPACEMIT_EP_INTRA_THREAD_AFFINITY': affinity,
                  'SPACEMIT_EP_DENSE_ACCURACY_LEVEL': '2',
                  'SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE': '1',
                  'SPACEMIT_EP_DISABLE_OP_TYPE_FILTER': CPU_OPS}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=options,
        providers=['SpaceMITExecutionProvider'], provider_options=[ep_options])
    session.disable_fallback()
    if 'SpaceMITExecutionProvider' not in session.get_providers():
        raise RuntimeError('AI Core provider failed to initialize; refusing silent CPU-only fallback')

    class Forward:
        def __call__(self, *tensors):
            names = ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')
            feeds = {k: v.detach().cpu().numpy() for k,v in zip(names,tensors)}
            return tuple(torch.from_numpy(v) for v in session.run(['logits','act_logits'], feeds))

    agent = Agent.__new__(Agent)
    agent.cfg = json.loads((Path(model_dir) / 'rl_agent_config.json').read_text())
    agent.tok = AutoTokenizer.from_pretrained(str(Path(model_dir) / 'tokenizer'), local_files_only=True)
    agent.device, agent.dtype = torch.device('cpu'), torch.float32
    agent.temperature = agent.cfg.get('temperature', [1.,1.,1.])
    agent.temperature_by_options = agent.cfg.get('temperature_by_options', {})
    agent.model = Forward()
    return agent, {'engine':'ONNX SpaceMIT AI Core + CPU', 'display_engine':'AI CORE / FP16',
                   'onnxruntime_version':ort.__version__, 'ai_core_affinity':affinity,
                   'onnx_model':str(path), 'onnx_file_dtype':'FP16',
                   'providers':session.get_providers(), 'ep_options':ep_options,
                   'optimization':'Fixed fast A100 FP16 offload; CPU Transpose compatibility; no graph switching'}
