"""Use ModernBERT's normal eager CPU path on Python 3.14 / PyTorch 2.9.

Transformers 5.0 decorates unused CPU methods at import time with torch.compile,
which PyTorch 2.9 rejects on Python 3.14 before its disable flag is examined.
Only bypass those decorators during the ModernBERT import; restore torch.compile
immediately. No system packages, weights, or model arithmetic are modified.
"""
import sys
import torch


def prepare():
    if sys.version_info >= (3, 14) and torch.__version__.startswith("2.9."):
        original_compile = torch.compile
        try:
            torch.compile = lambda model=None, **kwargs: model if model is not None else (lambda fn: fn)
            from transformers.models.modernbert import modeling_modernbert  # noqa: F401
        finally:
            torch.compile = original_compile

