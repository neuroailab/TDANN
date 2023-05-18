from types import SimpleNamespace

import torch
from classy_vision.generic.distributed_util import all_reduce_mean
from classy_vision.tasks import ClassyTask
from vissl.hooks import SSLClassyHookFunctions
from vissl.trainer.train_steps import register_train_step
from vissl.trainer.train_steps.standard_train_step import construct_sample_for_model
from vissl.utils.perf_stats import PerfTimer

from spacetorch.losses.losses_torch import spatial_correlation_loss

LastBatchInfo = SimpleNamespace


@register_train_step("custom_train_step")
def custom_train_step(task):
    """
    Single training iteration loop of the model.

    Performs: data read, forward, loss computation, backward, optimizer step, parameter
        updates.

    Various intermediate steps are also performed:
    - logging the training loss, training eta, LR, etc to loggers
    - logging to tensorboard,
    - performing any self-supervised method specific operations (like in MoCo approach,
        the momentum encoder is updated), computing the scores in swav
    - checkpointing model if user wants to checkpoint in the middle
    of an epoch
    """
    assert isinstance(task, ClassyTask), "task is not instance of ClassyTask"

    # reset the last batch info at every step
    task.last_batch = LastBatchInfo()

    # We'll time train_step and some of its sections, and accumulate values
    # into perf_stats if it were defined in local_variables:
    perf_stats = task.perf_stats
    timer_train_step = PerfTimer("train_step_total", perf_stats)
    timer_train_step.start()

    with PerfTimer("read_sample", perf_stats):
        sample = next(task.data_iterator)

    sample = construct_sample_for_model(sample, task)

    # Only need gradients during training
    grad_context = torch.enable_grad() if task.train else torch.no_grad()

    with grad_context:
        with PerfTimer("forward", perf_stats):
            model_output = task.model(sample["input"])

        task.last_batch.sample = sample
        task.last_batch.model_output = model_output
        target = sample["target"]

        # increment iteration by 1 (normally this would happen in the on_forward hook,
        # which I'm avoiding using here since it rigidly assumes model outputs are
        # torch.Tensors
        # task.iteration += 1

        # compute loss on this replica
        with PerfTimer("loss_compute", perf_stats):
            local_loss = task.loss(model_output, target)

        # compute spatial loss on this replica, not worried about aggregating for now
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, spatial_outputs = model_output

        spatial_losses = {}
        for layer, layer_output in spatial_outputs.items():
            features, pos = layer_output
            spatial_losses[layer] = spatial_correlation_loss(
                features.to(device),
                pos.coordinates.to(device),
                pos.neighborhood_indices.to(device),
            )

        task.last_batch.spatial_loss = spatial_losses

        # Reduce the loss value across all nodes and gpus.
        with PerfTimer("loss_all_reduce", perf_stats):
            loss = local_loss.detach().clone()
            task.last_batch.loss = all_reduce_mean(loss)

        task.losses.append(task.last_batch.loss.data.cpu().item() * target.size(0))

        # Update meters
        if len(task.meters) > 0 and (
            (task.train and task.config["METERS"]["enable_training_meter"])
            or (not task.train)
        ):
            with PerfTimer("meters_update", perf_stats):
                feature_output, _ = model_output
                feature_output = feature_output.cpu()

                for meter in task.meters:
                    meter.update(feature_output, target.detach().cpu())

        task.last_batch.target = target
        task.run_hooks(SSLClassyHookFunctions.on_loss_and_meter.name)

    # Run backward now and update the optimizer
    if task.train:
        with PerfTimer("backward", perf_stats):
            task.optimizer.zero_grad()
            local_loss.backward()

            task.run_hooks(SSLClassyHookFunctions.on_backward.name)

        with PerfTimer("optimizer_step", perf_stats):
            # Stepping the optimizer also updates learning rate, momentum etc
            # according to the schedulers (if any).
            task.optimizer.step(where=task.where)

        task.run_hooks(SSLClassyHookFunctions.on_update.name)
        task.num_updates += task.get_global_batchsize()

    timer_train_step.stop()
    timer_train_step.record()

    return task
