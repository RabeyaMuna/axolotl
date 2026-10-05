"""StableMax integration entry point."""

from axolotl.integrations.base import BasePlugin

from .args import StableMaxArgs  # noqa: F401
from .stablemax import stablemax_cross_entropy


class StableMaxPlugin(BasePlugin):
    """Plugin for StableMax integration with Axolotl."""

    def get_input_args(self):
        return "axolotl.integrations.stablemax.StableMaxArgs"

    def pre_model_load(self, cfg):
        """Patch the loss function to use StableMax cross-entropy if enabled."""
        if getattr(cfg, "stablemax", False):
            self._check_cross_entropy_conflicts(cfg)
            import torch.nn.functional as F

            F.cross_entropy = stablemax_cross_entropy

    def _check_cross_entropy_conflicts(self, cfg):
        """Check for conflicts with other integrations that patch cross_entropy."""
        conflicts = []
        if getattr(cfg, "liger_cross_entropy", False):
            conflicts.append("Liger cross_entropy")
        if getattr(cfg, "liger_fused_linear_cross_entropy", False):
            conflicts.append("Liger fused_linear_cross_entropy")
        if getattr(cfg, "cut_cross_entropy", False):
            conflicts.append("CutCrossEntropy")
        if conflicts:
            raise ValueError(
                f"StableMax cannot be enabled simultaneously with other cross-entropy patches: {', '.join(conflicts)}. Please disable one of these integrations to avoid runtime conflicts."
            )
