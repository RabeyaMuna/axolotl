"""
extra axolotl specific training args
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type

from transformers import TrainingArguments
from trl import CPOConfig, KTOConfig, ORPOConfig, PRMConfig, RewardConfig

from axolotl.integrations.config import merge_training_args

AxolotlTrainingMixins: Type = merge_training_args()


def _normalize_axolotl_training_args(args):
    if getattr(args, "max_seq_length", None) is None:
        args.max_seq_length = 2048
    if getattr(args, "sample_packing_efficiency", None) is None:
        args.sample_packing_efficiency = 1.0


@dataclass
class AxolotlTrainingArguments(AxolotlTrainingMixins, TrainingArguments):
    """
    Training arguments for Causal trainer

    This code is duplicated due to HF TrainingArguments not setting output_dir with a
    default value so it can't be used as a mixin.
    """

    def __post_init__(self):
        _normalize_axolotl_training_args(self)
        super().__post_init__()
        _normalize_axolotl_training_args(self)


@dataclass
class AxolotlORPOConfig(AxolotlTrainingMixins, ORPOConfig):
    """
    ORPO config for ORPO training
    """

    def __post_init__(self):
        _normalize_axolotl_training_args(self)
        super().__post_init__()
        _normalize_axolotl_training_args(self)


@dataclass
class AxolotlKTOConfig(AxolotlTrainingMixins, KTOConfig):
    """
    KTO config for KTO training
    """

    def __post_init__(self):
        _normalize_axolotl_training_args(self)
        super().__post_init__()
        _normalize_axolotl_training_args(self)


@dataclass
class AxolotlCPOConfig(AxolotlTrainingMixins, CPOConfig):
    """
    CPO config for CPO training
    """

    simpo_gamma: Optional[float] = field(
        default=None,
        metadata={"help": "simpo gamma parameter"},
    )

    def __post_init__(self):
        _normalize_axolotl_training_args(self)
        super().__post_init__()
        _normalize_axolotl_training_args(self)


@dataclass
class AxolotlRewardConfig(AxolotlTrainingMixins, RewardConfig):
    """
    Reward config for Reward training
    """

    def __post_init__(self):
        _normalize_axolotl_training_args(self)
        super().__post_init__()
        _normalize_axolotl_training_args(self)


@dataclass
class AxolotlPRMConfig(AxolotlTrainingMixins, PRMConfig):
    """
    PRM config for PRM training
    """

    def __post_init__(self):
        _normalize_axolotl_training_args(self)
        super().__post_init__()
        _normalize_axolotl_training_args(self)
