"""Restoring scheduler counters must also restore the first resumed step's LR."""

import copy
import math

import pytest
import torch

from speculators.train.checkpointer import BaseCheckpointer


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("tensor_lr", [False, True])
def test_scheduler_resume_restores_lr_and_next_step_trajectory(
    tmp_path, schedule, count, tensor_lr
):
    def decay(step):
        if schedule == "linear":
            return 1 - step / 20
        return (1 + math.cos(math.pi * step / 20)) / 2

    original_opts, original_schedulers = [], []
    for index in range(count):
        parameter = torch.nn.Parameter(torch.ones(1))
        lr = (index + 1) * 0.01
        opt = torch.optim.SGD([parameter], lr=torch.tensor(lr) if tensor_lr else lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, decay)
        original_opts.append(opt)
        original_schedulers.append(sched)
    for _ in range(7):
        for opt, sched in zip(original_opts, original_schedulers, strict=True):
            opt.step()
            sched.step()
    saved_opts = [copy.deepcopy(opt.state_dict()) for opt in original_opts]
    saved_schedulers = [
        copy.deepcopy(sched.state_dict()) for sched in original_schedulers
    ]
    epoch = tmp_path / "0"
    epoch.mkdir()
    payload = saved_schedulers[0] if count == 1 else saved_schedulers
    torch.save(payload, epoch / "scheduler_state_dict.pt")

    resumed_opts, resumed_schedulers = [], []
    for state in saved_opts:
        parameter = torch.nn.Parameter(torch.ones(1))
        lr = torch.tensor(0.1) if tensor_lr else 0.1
        opt = torch.optim.SGD([parameter], lr=lr)
        opt.load_state_dict(state)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, decay, last_epoch=0)
        resumed_opts.append(opt)
        resumed_schedulers.append(sched)
    assert float(resumed_opts[0].param_groups[0]["lr"]) != pytest.approx(
        float(original_opts[0].param_groups[0]["lr"])
    )
    checkpointer = BaseCheckpointer(tmp_path)
    checkpointer.load_scheduler_state_dict(
        resumed_schedulers[0] if count == 1 else resumed_schedulers
    )
    for original, resumed in zip(original_opts, resumed_opts, strict=True):
        assert float(resumed.param_groups[0]["lr"]) == pytest.approx(
            float(original.param_groups[0]["lr"])
        )
        assert isinstance(resumed.param_groups[0]["lr"], torch.Tensor) == tensor_lr
    for _ in range(3):
        for original, resumed, original_sched, resumed_sched in zip(
            original_opts,
            resumed_opts,
            original_schedulers,
            resumed_schedulers,
            strict=True,
        ):
            original.step()
            resumed.step()
            original_sched.step()
            resumed_sched.step()
            assert float(resumed.param_groups[0]["lr"]) == pytest.approx(
                float(original.param_groups[0]["lr"])
            )


def test_missing_scheduler_file_keeps_legacy_behavior(tmp_path):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    (tmp_path / "0").mkdir()
    BaseCheckpointer(tmp_path).load_scheduler_state_dict(scheduler)
    assert opt.param_groups[0]["lr"] == 0.1
