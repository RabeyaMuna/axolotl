"""Click CLI definitions for various axolotl commands."""

# pylint: disable=redefined-outer-name

import os
import subprocess  # nosec B404
from typing import Literal, Optional

import click
from dotenv import load_dotenv

import axolotl
from axolotl.cli.args import (
    EvaluateCliArgs,
    PreprocessCliArgs,
    QuantizeCliArgs,
    TrainerCliArgs,
    VllmServeCliArgs,
)
from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.utils import (
    add_options_from_config,
    add_options_from_dataclass,
    build_command,
    fetch_from_github,
    filter_none_kwargs,
    generate_config_files,
    launch_training,
)
from axolotl.utils import patch_optimized_env
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.config import AxolotlInputConfig

LOG = get_logger(__name__)

LAUNCHER_COMMAND_MAPPING = {
    "accelerate": ["accelerate", "launch"],
    "torchrun": ["torchrun"],
}


@click.group()
@click.version_option(version=axolotl.__version__, prog_name="axolotl")
def cli():
    """Axolotl CLI - Train and fine-tune large language models"""
    print_axolotl_text_art()
    load_dotenv()
    patch_optimized_env()


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option("--cloud", default=None, type=click.Path(exists=True, path_type=str))
@add_options_from_dataclass(PreprocessCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
def preprocess(config: str, cloud: Optional[str] = None, **kwargs):
    """
    Preprocess datasets before training.

    Args:
        config: Path to `axolotl` config YAML file.
        cloud: Path to a cloud accelerator configuration file.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """

    if cloud:
        from axolotl.cli.cloud import do_cli_preprocess

        do_cli_preprocess(cloud_config=cloud, config=config)
    else:
        from axolotl.cli.preprocess import do_cli

        do_cli(config=config, **kwargs)


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--launcher",
    type=click.Choice(["accelerate", "torchrun", "python"]),
    default="accelerate",
    help="Launcher to use for multi-GPU training",
)
@click.option("--cloud", default=None, type=click.Path(exists=True, path_type=str))
@click.option(
    "--sweep",
    type=click.Path(exists=True, path_type=str),
    help="YAML config for sweeping hyperparameters",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
@click.pass_context
def train(
    ctx: click.Context,
    config: str,
    launcher: Literal["accelerate", "torchrun", "python"] = "accelerate",
    cloud: str | None = None,
    sweep: str | None = None,
    **kwargs,
):
    """
    Train or fine-tune a model.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU training ("accelerate", "torchrun", or "python").
        cloud: Path to a cloud accelerator configuration file
        sweep: Path to YAML config for sweeping hyperparameters.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Extract launcher args from extra args (after --)
    launcher_args = ctx.args if ctx.args else []

    # Handle Ray launcher override
    _launcher = None if kwargs.get("use_ray") else launcher

    # Process each configuration
    for cfg_file in generate_config_files(config, sweep):
        try:
            launch_training(cfg_file, _launcher, cloud, kwargs, launcher_args)
        except subprocess.CalledProcessError as exc:
            LOG.error(f"Failed to train/fine-tune config '{cfg_file}': {exc}")
            if not sweep:
                raise exc
        finally:
            # Only delete temp files, not the original config
            if cfg_file != config:
                os.unlink(cfg_file)


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--launcher",
    type=click.Choice(["accelerate", "torchrun", "python"]),
    default="accelerate",
    help="Launcher to use for multi-GPU evaluation",
)
@add_options_from_dataclass(EvaluateCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
@click.pass_context
def evaluate(ctx: click.Context, config: str, launcher: str, **kwargs):
    """
    Evaluate a model.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU evaluation ("accelerate", "torchrun", or "python").
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Extract launcher args from extra args (after --)
    launcher_args = ctx.args if ctx.args else []

    if launcher in LAUNCHER_COMMAND_MAPPING:
        base_cmd = (
            LAUNCHER_COMMAND_MAPPING[launcher]
            + launcher_args
            + ["-m", "axolotl.cli.evaluate"]
        )
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.evaluate import do_cli

        do_cli(config=config, **kwargs)


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--launcher",
    type=click.Choice(["accelerate", "torchrun", "python"]),
    default="accelerate",
    help="Launcher to use for multi-GPU inference",
)
@click.option("--gradio", is_flag=True, help="Launch Gradio interface")
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
@click.pass_context
def inference(ctx: click.Context, config: str, launcher: str, gradio: bool, **kwargs):
    """
    Run inference with a trained model.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU inference ("accelerate", "torchrun", or "python").
        gradio: Whether to use Gradio browser interface or command line for inference.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Extract launcher args from extra args (after --)
    launcher_args = ctx.args if ctx.args else []

    if launcher in LAUNCHER_COMMAND_MAPPING:
        base_cmd = (
            LAUNCHER_COMMAND_MAPPING[launcher]
            + launcher_args
            + ["-m", "axolotl.cli.inference"]
        )
        if config:
            base_cmd.append(config)
        if gradio:
            base_cmd.append("--gradio")
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.inference import do_cli

        do_cli(config=config, gradio=gradio, **kwargs)


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--launcher",
    type=click.Choice(["accelerate", "torchrun", "python"]),
    default="accelerate",
    help="Launcher to use for weight merging",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
@click.pass_context
def merge_sharded_fsdp_weights(
    ctx: click.Context, config: str, launcher: str, **kwargs
):
    """
    Merge sharded FSDP model weights.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for weight merging ("accelerate", "torchrun", or "python").
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Extract launcher args from extra args (after --)
    launcher_args = ctx.args if ctx.args else []

    if launcher in LAUNCHER_COMMAND_MAPPING:
        base_cmd = (
            LAUNCHER_COMMAND_MAPPING[launcher]
            + launcher_args
            + ["-m", "axolotl.cli.merge_sharded_fsdp_weights"]
        )
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.merge_sharded_fsdp_weights import do_cli

        do_cli(config=config, **kwargs)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
def merge_lora(config: str, **kwargs):
    """
    Merge trained LoRA adapters into a base model.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    from axolotl.cli.merge_lora import do_cli

    do_cli(config=config, **kwargs)


@cli.command()
@click.argument("directory", type=click.Choice(["examples", "deepspeed_configs"]))
@click.option("--dest", help="Destination directory")
def fetch(directory: str, dest: Optional[str]):
    """
    Fetch example configs or other resources.

    Available directories:
    - examples: Example configuration files
    - deepspeed_configs: DeepSpeed configuration files

    Args:
        directory: One of `examples`, `deepspeed_configs`.
        dest: Optional destination directory.
    """
    fetch_from_github(f"{directory}/", dest)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@add_options_from_dataclass(VllmServeCliArgs)
@filter_none_kwargs
def vllm_serve(config: str, **cli_args: VllmServeCliArgs):
    from axolotl.cli.vllm_serve import do_vllm_serve

    do_vllm_serve(config, cli_args)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@add_options_from_dataclass(QuantizeCliArgs)
@filter_none_kwargs
def quantize(config: str, **cli_args: QuantizeCliArgs):
    from axolotl.cli.quantize import do_quantize

    do_quantize(config, cli_args)


@cli.command(name="lm-eval")
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option("--cloud", default=None, type=click.Path(exists=True, path_type=str))
def lm_eval(config: str, cloud: Optional[str] = None):
    """
    Use lm-evaluation-harness to evaluate a trained language model.
    """
    if cloud:
        from axolotl.cli.cloud import do_cli_lm_eval

        do_cli_lm_eval(cloud_config=cloud, config=config)
        return

    import yaml

    from axolotl.utils.dict import DictDefault

    with open(config, encoding="utf-8") as file:
        cfg: DictDefault = DictDefault(yaml.safe_load(file))

    for lm_eval_args in _build_lm_eval_command(
        cfg.lm_eval_tasks,
        bfloat16=cfg.bfloat16 or cfg.bf16,
        flash_attention=cfg.flash_attention,
        output_dir=cfg.output_dir,
        batch_size=cfg.lm_eval_batch_size,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        wandb_name=cfg.wandb_name,
        model=cfg.lm_eval_model or cfg.hub_model_id,
        revision=cfg.revision,
        apply_chat_template=cfg.apply_chat_template,
        fewshot_as_multiturn=cfg.fewshot_as_multiturn,
    ):
        subprocess.run(lm_eval_args, check=True)  # nosec B603


def _build_lm_eval_command(
    tasks: list[str],
    bfloat16=True,
    flash_attention=False,
    output_dir="./",
    batch_size=8,
    wandb_project=None,
    wandb_entity=None,
    wandb_name=None,
    model=None,
    revision=None,
    apply_chat_template=None,
    fewshot_as_multiturn=None,
):
    from collections import defaultdict
    from datetime import datetime

    tasks_by_num_fewshot: dict[str, list] = defaultdict(list)
    if isinstance(tasks, str):
        tasks = [tasks]
    for task in tasks:
        num_fewshot = "-1"
        task_parts = task.split(":")
        task_name = task_parts[0]
        if len(task_parts) == 2:
            task_name, num_fewshot = task_parts
        tasks_by_num_fewshot[str(num_fewshot)].append(task_name)

    for num_fewshot, tasks_list in tasks_by_num_fewshot.items():
        tasks_str = ",".join(tasks_list)
        num_fewshot_val = num_fewshot if num_fewshot != "-1" else None
        pretrained = "pretrained="
        pretrained += model if model else output_dir
        fa2 = ",attn_implementation=flash_attention_2" if flash_attention else ""
        dtype = ",dtype=bfloat16" if bfloat16 else ",dtype=float16"
        revision = f",revision={revision}" if revision else ""
        output_path = output_dir
        output_path += "" if output_dir.endswith("/") else "/"
        output_path += "lm_eval_results/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        lm_eval_args = [
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"{pretrained}{fa2}{dtype}{revision}",
            "--tasks",
            tasks_str,
            "--batch_size",
            str(batch_size),
            "--output_path",
            output_path,
        ]
        wandb_args = []
        if wandb_project:
            wandb_args.append(f"project={wandb_project}")
        if wandb_entity:
            wandb_args.append(f"entity={wandb_entity}")
        if wandb_name:
            wandb_args.append(f"name={wandb_name}")
        if wandb_args:
            lm_eval_args.append("--wandb_args")
            lm_eval_args.append(",".join(wandb_args))
        if apply_chat_template:
            lm_eval_args.append("--apply_chat_template")
        if num_fewshot_val:
            lm_eval_args.append("--num_fewshot")
            lm_eval_args.append(str(num_fewshot_val))
            if apply_chat_template and fewshot_as_multiturn:
                lm_eval_args.append("--fewshot_as_multiturn")

        yield lm_eval_args


@cli.command()
@click.argument("model", type=click.Path(exists=True, path_type=str))
@click.argument("output", type=click.Path(exists=False, path_type=str))
def delinearize_llama4(model: str, output: str):
    from axolotl.cli.delinearize_llama4 import do_cli as do_delinearize_llama4

    do_delinearize_llama4(model, output)


def main():
    cli()


if __name__ == "__main__":
    main()
