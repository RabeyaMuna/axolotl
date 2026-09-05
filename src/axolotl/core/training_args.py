"""
extra axolotl specific training args
"""

from dataclasses import dataclass, field

from PIL.Image import Resampling
from transformers import TrainingArguments
from trl import CPOConfig, KTOConfig, ORPOConfig, PRMConfig, RewardConfig


@dataclass
class AxolotlTrainingMixins:
    """
    Mixin class for the Axolotl training args.
    """

    # pylint: disable=duplicate-code
    model_type: str | None = field(
        default=None, metadata={"help": "HF model configuration model_type."}
    )
    lr_quadratic_warmup: bool = field(
        default=False,
        metadata={"help": "Use quadratic warmup for cosine scheduling."},
    )
    pretraining: bool = field(
        default=False,
        metadata={
            "help": "Indicates to trainer whether we are doing continued pretraining."
        },
    )
    sample_packing: bool = field(
        default=False,
        metadata={"help": "Use sample packing for efficient training."},
    )
    sample_packing_sequentially: bool = field(
        default=False,
        metadata={
            "help": "Use next-fit sample packing that preserves the order of samples coming from the sampler. Use in combination with curriculum_sampling for fully sequential packing."
        },
    )
    multipack_real_batches: bool = field(
        default=False,
        metadata={"help": "Use real batches for efficient training."},
    )
    eval_sample_packing: bool | None = field(
        default=None,
        metadata={"help": "Use sample packing for efficient evals."},
    )
    sample_packing_efficiency: float = field(
        default=1.0,
        metadata={"help": "Sample packing efficiency for calculating batch length."},
    )
    sample_packing_bin_size: int = field(
        default=200,
        metadata={
            "help": "The max number of samples that packed sample can contain after packing. Increase for better packing."
        },
    )
    sample_packing_group_size: int = field(
        default=100000,
        metadata={
            "help": "The number of samples to group together for packing. Increase for better packing."
        },
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length the model can handle"},
    )
    relora_steps: int | None = field(
        default=None,
        metadata={"help": "how often to reset for ReLoRA"},
    )
    relora_warmup_steps: int | None = field(
        default=None,
        metadata={"help": "how many warmup steps to take after reset for ReLoRA"},
    )
    relora_anneal_steps: int | None = field(
        default=None,
        metadata={"help": "how many warmup steps to take after reset for ReLoRA"},
    )
    relora_prune_ratio: float | None = field(
        default=0.9,
        metadata={"help": "prune ratio for magnitude pruning of the optimizer"},
    )
    bench_split: str | None = field(
        default="eval", metadata={"help": "The benchmark split to run on"}
    )
    bench_dataset: str | None = field(
        default="pharaouk/dharma-1/dharma_1_mini.json",
        metadata={
            "help": "Benchmark dataset to use: options are `mmlu-zs`, `mmlu-fs`, or the full path to the dataset file"
        },
    )
    do_bench_eval: bool | None = field(
        default=False, metadata={"help": "Whether to run the Benchmark evaluation."}
    )
    do_causal_lm_eval: bool | None = field(
        default=False, metadata={"help": "Whether to run the Causal LM evaluation."}
    )
    max_bench_samples: int | None = field(
        default=None,
        metadata={
            "help": "If set, only evaluates on `max_bench_samples` of the benchmark dataset."
        },
    )
    bench_source_max_len: int = field(
        default=2048, metadata={"help": "Maximum source sequence length for bench."}
    )
    dataloader_prefetch_factor: int | None = field(
        default=None,
        metadata={"help": "prefetch_factor argument to the dataloader"},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for dataset loading"},
    )

    cosine_min_lr_ratio: float | None = field(
        default=None,
        metadata={"help": "Minimum learning rate is min_lr_ratio * learning_rate"},
    )
    cosine_constant_lr_ratio: float | None = field(
        default=None,
        metadata={
            "help": "Starting constant learning rate step is cosine_constant_lr_ratio * max_steps"
        },
    )
    loraplus_lr_ratio: float | None = field(
        default=None, metadata={"help": "loraplus learning rate ratio lr_B / lr_A."}
    )
    loraplus_lr_embedding: float | None = field(
        default=1e-6,
        metadata={"help": "loraplus learning rate for lora embedding layers."},
    )
    embedding_lr_scale: float | None = field(
        default=None,
        metadata={"help": "Scale the learning rate for the embedding layers."},
    )
    lr_groups: list[dict] | None = field(
        default=None,
        metadata={"help": "Specify learning rate groups for with different LRs."},
    )
    embedding_lr: float | None = field(
        default=None,
        metadata={"help": "absolute learning rate for the embedding layers."},
    )
    qlora: bool = field(
        default=False,
        metadata={"help": "whether this is a qlora training"},
    )
    orpo_alpha: float | None = field(
        default=None,
    )
    lisa_n_layers: int | None = field(
        default=None,
        metadata={"help": "the number of activate layers in LISA"},
    )
    lisa_step_interval: int | None = field(
        default=None,
        metadata={"help": "how often to switch layers in LISA"},
    )
    lisa_layers_attribute: str | None = field(
        default=None,
        metadata={"help": "path under the model to access the layers"},
    )
    curriculum_sampling: bool | None = field(
        default=None,
        metadata={"help": "whether to use sequential sampling for curriculum learning"},
    )
    alternate_lr_scheduler_type: str | None = field(
        default=None,
        metadata={
            "help": "workaround to pass an alternate lr scheduler to the HF trainer"
        },
    )
    chat_template: str | None = field(
        default=None,
        metadata={"help": "Chat template converting chat messages to text"},
    )

    kd_ce_alpha: float | None = field(
        default=None,
        metadata={
            "help": "The alpha scaling parameter for SFT cross entropy loss when using KD"
        },
    )

    kd_alpha: float | None = field(
        default=1.0,
        metadata={"help": "The alpha scaling parameter for KD loss"},
    )

    kd_temperature: float | None = field(
        default=1.0,
        metadata={
            "help": "the temperature parameter for KL divergence loss when using KD"
        },
    )

    kd_zscore_base_temp: float | None = field(
        default=None,
        metadata={
            "help": "the base temperature parameter for KL divergence with z-score when using KD"
        },
    )

    kd_top_k_before_softmax: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to apply top_k_before_softmax to the logits when using KD"
        },
    )

    adam_beta3: float | None = field(
        default=None,
        metadata={
            "help": "The beta3 hyperparameter used in some optimizers such as CAME"
        },
    )
    adam_epsilon2: float | None = field(
        default=None,
        metadata={
            "help": "The epsilon2 hyperparameter used in some optimizers such as CAME"
        },
    )

    # multi-modal section

    image_size: int | tuple[int, int] | None = field(
        default=None,
        metadata={"help": "The size of the image to resize to"},
    )

    image_resize_algorithm: Resampling | None = field(
        default=None,
        metadata={"help": "The algorithm to use for image resizing"},
    )

    # end of multi-modal section


@dataclass
class AxolotlTrainingArguments(AxolotlTrainingMixins, TrainingArguments):
    """
    Training arguments for Causal trainer

    This code is duplicated due to HF TrainingArguments not setting output_dir with a
    default value so it can't be used as a mixin.
    """


@dataclass
class AxolotlORPOConfig(AxolotlTrainingMixins, ORPOConfig):
    """
    ORPO config for ORPO training
    """


@dataclass
class AxolotlKTOConfig(AxolotlTrainingMixins, KTOConfig):
    """
    KTO config for KTO training
    """


@dataclass
class AxolotlCPOConfig(AxolotlTrainingMixins, CPOConfig):
    """
    CPO config for CPO training
    """

    simpo_gamma: float | None = field(
        default=None,
        metadata={"help": "simpo gamma parameter"},
    )


@dataclass
class AxolotlRewardConfig(AxolotlTrainingMixins, RewardConfig):
    """
    Reward config for Reward training
    """


@dataclass
class AxolotlPRMConfig(AxolotlTrainingMixins, PRMConfig):
    """
    PRM config for PRM training
    """
