from enum import auto, Enum
import logging
from typing import List

import torch

from classy_vision.hooks.classy_hook import ClassyHook
from classy_vision.generic.distributed_util import is_primary
from classy_vision import tasks

from vissl.config import AttrDict
from vissl.hooks.deepclusterv2_hooks import ClusterMemoryHook, InitMemoryHook  # noqa
from vissl.hooks.dino_hooks import DINOHook

# from vissl.hooks.ema_hooks import EmaHook
from vissl.hooks.grad_clip_hooks import GradClipHook  # noqa

# from vissl.hooks.ibot_hooks import IBOTHook
from vissl.hooks.log_hooks import (  # noqa
    DumpMemoryOnException,
    LogGpuMemoryHook,
    LogGpuStatsHook,
    LogLossLrEtaHook,
    LogLossMetricsCheckpointHook,
    LogPerfTimeMetricsHook,
)
from vissl.hooks.moco_hooks import MoCoHook  # noqa

# from vissl.hooks.model_output_mask_hook import ModelOutputMaskHook
# from vissl.hooks.profiling_hook import CudaSynchronizeHook
from vissl.hooks.profiling_hook import ProfilingHook
from vissl.hooks.state_update_hooks import (  # noqa
    FreezeParametersHook,
    SetDataSamplerEpochHook,
    SSLModelComplexityHook,
)
from vissl.hooks.swav_hooks import (  # noqa  # noqa
    NormalizePrototypesHook,
    SwAVUpdateQueueScoresHook,
)
from vissl.hooks.swav_momentum_hooks import (
    SwAVMomentumHook,
    SwAVMomentumNormalizePrototypesHook,
)
from vissl.hooks.tensorboard_hook import SSLTensorboardHook  # noqa
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.tensorboard import is_tensorboard_available, get_tensorboard_dir

# define custom spatial tensorboard hooks
BYTE_TO_MiB = 2**20


class SSLClassyHookFunctions(Enum):
    """
    Enumeration of all the hook functions in the ClassyHook class.
    """

    on_start = auto()
    on_phase_start = auto()
    on_forward = auto()
    on_loss_and_meter = auto()
    on_backward = auto()
    on_update = auto()
    on_step = auto()
    on_phase_end = auto()
    on_end = auto()
    on_exception = auto()


def add_loss_hooks(hooks, loss_cfg, cfg):
    if cfg.LOSS.name == "swav_loss":
        hooks.extend([SwAVUpdateQueueScoresHook(), NormalizePrototypesHook()])
    if cfg.LOSS.name == "swav_momentum_loss":
        hooks.extend(
            [
                SwAVMomentumHook(
                    cfg.LOSS["swav_momentum_loss"]["momentum"],
                    cfg.LOSS["swav_momentum_loss"]["momentum_eval_mode_iter_start"],
                    cfg.LOSS["swav_momentum_loss"]["crops_for_assign"],
                ),
                SwAVMomentumNormalizePrototypesHook(),
            ]
        )
    if cfg.LOSS.name == "dino_loss":
        hooks.append(DINOHook())
    # if cfg.LOSS.name in {"ibot_loss"}:
    #     hooks.append(IBOTHook())
    if cfg.LOSS.name == "deepclusterv2_loss":
        hooks.extend([InitMemoryHook(), ClusterMemoryHook()])
    if cfg.LOSS.name == "moco_loss":
        hooks.extend(
            [
                MoCoHook(
                    cfg.LOSS["moco_loss"]["momentum"],
                    shuffle_batch=(not cfg.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN),
                )
            ]
        )
    return hooks


class SpatialSSLTensorboardHook(SSLTensorboardHook):
    """
    Inherit from SSLTensorboardHook, then overload just the bits we need
    """

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Called after every parameters update if tensorboard hook is enabled.
        Logs the parameter gradients if they are being set to log,
        log the scalars like training loss, learning rate, average training
        iteration time, batch size per gpu, img/sec/gpu, ETA, gpu memory used,
        peak gpu memory used.
        """

        if not is_primary():
            return

        iteration = task.iteration

        if (
            self.log_params_every_n_iterations > 0
            and self.log_params_gradients
            and task.train
            and iteration % self.log_params_every_n_iterations == 0
        ):
            logging.info(f"Logging Parameter gradients. Iteration {iteration}")
            for name, parameter in task.base_model.named_parameters():
                if parameter.grad is not None:
                    try:
                        self.tb_writer.add_histogram(
                            f"Gradients/{name}",
                            parameter.grad,
                            global_step=task.iteration,
                        )
                    except ValueError:
                        logging.info(
                            f"Gradient histogram empty for {name}, "
                            f"iteration {task.iteration}. Unable to "
                            f"log gradient."
                        )

        if iteration % task.config["LOG_FREQUENCY"] == 0 or (
            iteration <= 100 and iteration % 5 == 0
        ):
            logging.info(f"Logging metrics. Iteration {iteration}")
            self.tb_writer.add_scalar(
                tag="Training/Loss",
                scalar_value=round(task.last_batch.loss.data.cpu().item(), 5),
                global_step=iteration,
            )

            self.tb_writer.add_scalar(
                tag="Training/Learning_rate",
                scalar_value=round(task.optimizer.options_view.lr, 5),
                global_step=iteration,
            )

            if hasattr(task.last_batch, "spatial_loss"):
                for layer, loss in task.last_batch.spatial_loss.items():
                    self.tb_writer.add_scalar(
                        tag=f"Training/Spatial_loss/{layer}",
                        scalar_value=round(loss.data.cpu().item(), 5),
                        global_step=iteration,
                    )

            # Batch processing time
            if len(task.batch_time) > 0:
                batch_times = task.batch_time
            else:
                batch_times = [0]

            batch_time_avg_s = sum(batch_times) / max(len(batch_times), 1)
            self.tb_writer.add_scalar(
                tag="Speed/Batch_processing_time_ms",
                scalar_value=int(1000.0 * batch_time_avg_s),
                global_step=iteration,
            )

            # Images per second per replica
            pic_per_batch_per_gpu = task.config["DATA"]["TRAIN"][
                "BATCHSIZE_PER_REPLICA"
            ]
            pic_per_batch_per_gpu_per_sec = (
                int(pic_per_batch_per_gpu / batch_time_avg_s)
                if batch_time_avg_s > 0
                else 0.0
            )
            self.tb_writer.add_scalar(
                tag="Speed/img_per_sec_per_gpu",
                scalar_value=pic_per_batch_per_gpu_per_sec,
                global_step=iteration,
            )

            # ETA
            avg_time = sum(batch_times) / len(batch_times)
            eta_secs = avg_time * (task.max_iteration - iteration)
            self.tb_writer.add_scalar(
                tag="Speed/ETA_hours",
                scalar_value=eta_secs / 3600.0,
                global_step=iteration,
            )

            # GPU Memory
            if torch.cuda.is_available():
                # Memory actually being used
                self.tb_writer.add_scalar(
                    tag="Memory/Peak_GPU_Memory_allocated_MiB",
                    scalar_value=torch.cuda.max_memory_allocated() / BYTE_TO_MiB,
                    global_step=iteration,
                )

                # Memory reserved by PyTorch's memory allocator
                self.tb_writer.add_scalar(
                    tag="Memory/Peak_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.max_memory_reserved()
                    / BYTE_TO_MiB,  # byte to MiB
                    global_step=iteration,
                )

                self.tb_writer.add_scalar(
                    tag="Memory/Current_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.memory_reserved()
                    / BYTE_TO_MiB,  # byte to MiB
                    global_step=iteration,
                )


def get_spatial_tensorboard_hook(cfg):
    from torch.utils.tensorboard import SummaryWriter

    tensorboard_dir = get_tensorboard_dir(cfg)
    flush_secs = cfg.TENSORBOARD_SETUP.FLUSH_EVERY_N_MIN * 60
    return SpatialSSLTensorboardHook(
        tb_writer=SummaryWriter(log_dir=tensorboard_dir, flush_secs=flush_secs),
        log_params=cfg.TENSORBOARD_SETUP.LOG_PARAMS,
        log_params_every_n_iterations=cfg.TENSORBOARD_SETUP.LOG_PARAMS_EVERY_N_ITERS,
        log_params_gradients=cfg.TENSORBOARD_SETUP.LOG_PARAMS_GRADIENTS,
        log_activation_statistics=cfg.MONITORING.MONITOR_ACTIVATION_STATISTICS,
    )


def spatial_hook_generator(cfg: AttrDict) -> List[ClassyHook]:
    hooks = []

    # conditionally add hooks based on use-case
    if cfg.HOOKS.PERF_STATS.MONITOR_PERF_STATS:
        perf_stat_freq = (
            cfg.HOOKS.PERF_STATS.PERF_STAT_FREQUENCY
            if cfg.HOOKS.PERF_STATS.PERF_STAT_FREQUENCY > 0
            else None
        )
        hooks.append(LogPerfTimeMetricsHook(perf_stat_freq))

    # add the loss hooks based on the loss being used
    hooks = add_loss_hooks(hooks, cfg.LOSS, cfg)

    if cfg.HOOKS.MODEL_COMPLEXITY.COMPUTE_COMPLEXITY:
        hooks.extend([SSLModelComplexityHook()])
    if cfg.HOOKS.LOG_GPU_STATS:
        hooks.extend([LogGpuStatsHook()])
    if cfg.HOOKS.MEMORY_SUMMARY.PRINT_MEMORY_SUMMARY:
        hooks.extend([LogGpuMemoryHook(cfg.HOOKS.MEMORY_SUMMARY.LOG_ITERATION_NUM)])
    if cfg.HOOKS.MEMORY_SUMMARY.DUMP_MEMORY_ON_EXCEPTION:
        hooks.append(DumpMemoryOnException())
    if cfg.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD:
        assert is_tensorboard_available(), (
            "Tensorboard must be installed to use it. Please install tensorboard using:"
            "If pip environment: `pip install tensorboard` "
            "If using conda and you prefer conda install of tensorboard: "
            "`conda install -c conda-forge tensorboard`"
        )
        tb_hook = get_spatial_tensorboard_hook(cfg)
        hooks.extend([tb_hook])
    if cfg.MODEL.GRAD_CLIP.USE_GRAD_CLIP:
        hooks.extend(
            [
                GradClipHook(
                    norm_type=cfg.MODEL.GRAD_CLIP.NORM_TYPE,
                    max_norm=cfg.MODEL.GRAD_CLIP.MAX_NORM,
                )
            ]
        )

    # hooks that are used irrespective of workflow type
    rolling_btime_freq = (
        cfg.HOOKS.PERF_STATS.ROLLING_BTIME_FREQ
        if cfg.HOOKS.PERF_STATS.ROLLING_BTIME_FREQ > 0
        else None
    )

    # if CudaSynchronizeHook.is_enabled(cfg.MODEL):
    #     hooks.append(CudaSynchronizeHook())

    if ProfilingHook.is_enabled(cfg.PROFILING):
        hooks.append(ProfilingHook(profiling_config=cfg.PROFILING))

    world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    checkpoint_folder = get_checkpoint_folder(cfg)

    hooks.extend(
        [
            SetDataSamplerEpochHook(),
            FreezeParametersHook(),
            LogLossMetricsCheckpointHook(world_size),
            LogLossLrEtaHook(checkpoint_folder, rolling_btime_freq),
        ]
    )

    return hooks
