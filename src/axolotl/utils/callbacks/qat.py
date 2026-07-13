"""QAT Callback for HF Causal Trainer"""

from functools import lru_cache, partial

from torch import nn
from transformers import TrainerCallback

from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.quantization import QATConfig

LOG = get_logger(__name__)


@lru_cache(maxsize=1)
def _fake_quantized_modules():
    try:
        from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
        from torchao.quantization.qat.linear import FakeQuantizedLinear
    except ImportError as exc:
        raise ImportError("QAT requires torchao to be installed.") from exc

    return FakeQuantizedLinear, FakeQuantizedEmbedding


def toggle_fake_quant(mod: nn.Module, enable: bool):
    """
    Toggle fake quantization for any fake quantized linear or embedding layers in the model.

    Args:
        mod: The module to toggle fake quantization for.
        enable: Whether to enable or disable fake quantization.
    """
    fake_quantized_linear, fake_quantized_embedding = _fake_quantized_modules()
    if isinstance(mod, (fake_quantized_linear, fake_quantized_embedding)):
        if (
            isinstance(mod, fake_quantized_linear)
            and mod.activation_fake_quantizer is not None
        ):
            mod.activation_fake_quantizer.enabled = enable
        mod.weight_fake_quantizer.enabled = enable


class QATCallback(TrainerCallback):
    """
    Callback to toggle fake quantization for the model.
    """

    def __init__(self, cfg: QATConfig):
        self.cfg = cfg

    def on_step_begin(
        self, args, state, control, model, **kwargs
    ):  # pylint: disable=unused-argument,arguments-differ
        if self.cfg.fake_quant_after_n_steps is not None:
            if state.global_step == 0:
                LOG.info(f"Disabling fake quantization at step {state.global_step}")
                model.apply(partial(toggle_fake_quant, enable=False))
            elif state.global_step == self.cfg.fake_quant_after_n_steps:
                LOG.info(f"Enabling fake quantization at step {state.global_step}")
                model.apply(partial(toggle_fake_quant, enable=True))
