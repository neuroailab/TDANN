from pathlib import Path
import os
import tempfile

import torch
import torch.nn as nn

from classy_vision.generic.util import load_and_broadcast_checkpoint
from hydra.experimental import compose, initialize_config_module
from vissl.data import build_dataset, build_dataloader
from vissl.utils.hydra_config import convert_to_attrdict, AttrDict
from vissl.models import build_model
from vissl.utils.checkpoint import init_model_from_consolidated_weights
from vissl.losses.simclr_info_nce_loss import SimclrInfoNCECriterion

from spacetorch.paths import (
    POSITION_DIR,
    DEFAULT_IMAGENET_VAL_DIR,
    git_root,
)


def _restore_model_weights(config, model, path):
    load_and_broadcast_checkpoint(path, device=torch.device("cpu"))

    params_from_file = config["MODEL"]["WEIGHTS_INIT"]
    skip_layers = params_from_file.get("SKIP_LAYERS", [])
    replace_prefix = params_from_file.get("REMOVE_PREFIX", None)
    append_prefix = params_from_file.get("APPEND_PREFIX", None)
    state_dict_key_name = params_from_file.get("STATE_DICT_KEY_NAME", None)

    # we initialize the weights from this checkpoint. However, we
    # don't care about the other metadata like iteration number etc.
    # So the method only reads the state_dict
    init_model_from_consolidated_weights(
        config,
        model,
        state_dict=torch.load(path),
        state_dict_key_name=state_dict_key_name,
        skip_layers=skip_layers,
        replace_prefix=replace_prefix,
        append_prefix=append_prefix,
    )
    return model


def restore(config_path, ckpt_path):
    os.chdir(git_root)

    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=[f"config={config_path}"])

    with tempfile.TemporaryDirectory() as tmpdirname:
        cfg["config"]["CHECKPOINT"]["DIR"] = tmpdirname
        cfg["config"]["MODEL"]["TRUNK"]["TRUNK_PARAMS"]["position_dir"] = (
            POSITION_DIR
            / cfg["config"]["MODEL"]["TRUNK"]["TRUNK_PARAMS"]["position_dir"]
        )
        _, config = convert_to_attrdict(cfg)
        model = build_model(config["MODEL"], config["OPTIMIZER"])
        model = _restore_model_weights(config, model, ckpt_path)
        config.MODEL.FEATURE_EVAL_SETTINGS["EVAL_TRUNK_AND_HEAD"] = True

    return config, model


class TaskLossEval:
    def __init__(
        self,
        ckpt_path: Path,
        config_path: str,
        input_batch=None,
    ):
        self.ckpt_path = ckpt_path
        self.config_path = config_path

        self._build_model()
        if input_batch is None:
            self.input_batch = self._build_input_batch()
        else:
            self.input_batch = input_batch

        self._buffer_params = AttrDict(
            {"effective_batch_size": 256, "embedding_dim": 128, "world_size": 1}
        )
        self._temperature = 0.1
        self.info_criterion = SimclrInfoNCECriterion(
            self._buffer_params, self._temperature
        )

    def _build_model(self):
        self.cfg, self.model = restore(self.config_path, self.ckpt_path)
        self.model.to("cuda")

    def _build_input_batch(self):
        self.cfg.DATA.NUM_DATALOADER_WORKERS = 1
        self.cfg.DATA.TRAIN.DATA_PATHS = [str(DEFAULT_IMAGENET_VAL_DIR)]
        self.cfg.DATA.TRAIN.DATA_SOURCES = ["disk_folder"]
        self.cfg.DATA.TRAIN.DATA_LIMIT = 512
        dataset = build_dataset(self.cfg, split="TRAIN")
        dl = build_dataloader(
            dataset=dataset,
            dataset_config=self.cfg.DATA["TRAIN"],
            num_dataloader_workers=1,
            pin_memory=True,
            multi_processing_method="forkserver",
            device=torch.device("cuda"),
            split="TRAIN",
        )
        batch = next(iter(dl))
        return batch["data"][0]

    def __call__(self) -> float:
        flat_outputs, _ = self.model(self.input_batch)
        normalized_output = nn.functional.normalize(flat_outputs, dim=1, p=2)
        loss_tensor = self.info_criterion(normalized_output)

        return loss_tensor.detach().cpu().numpy().item()
