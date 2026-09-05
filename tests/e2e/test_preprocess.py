"""E2E Test the preprocess cli"""

from pathlib import Path

from accelerate.test_utils import execute_subprocess_async

from axolotl.utils.dict import DictDefault

from tests.utils.dataset_config import get_alpaca_dataset_config, write_config_to_yaml

AXOLOTL_ROOT = Path(__file__).parent.parent.parent


class TestPreprocess:
    """test cases for preprocess"""

    def test_w_deepspeed(self, temp_dir):
        """make sure preproces doesn't choke when using deepspeed in the config"""
        # Get shared dataset config
        dataset_config = get_alpaca_dataset_config()

        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen2.5-0.5B",
                "sequence_len": 2048,
                "val_set_size": 0.01,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "bf16": "auto",
                "deepspeed": str(AXOLOTL_ROOT / "deepspeed_configs/zero1.json"),
                "dataset_prepared_path": temp_dir + "/last_run_prepared",
            }
        )
        # Merge dataset config
        cfg.update(dataset_config)

        # Write config to yaml file
        config_path = write_config_to_yaml(cfg, temp_dir)

        execute_subprocess_async(
            [
                "axolotl",
                "preprocess",
                str(config_path),
            ]
        )

        assert (Path(temp_dir) / "last_run_prepared").exists()
