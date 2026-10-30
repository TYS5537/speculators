"""Real three-rank CPU DDP coverage, including non-power-of-two averaging."""

import copy
import inspect
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from speculators.models.metrics import resolve_training_loss
from speculators.train.ddp_options import resolve_ddp_options
from speculators.train.trainer import Trainer
from tests.unit.models.test_leagle import make_model
from tests.unit.models.test_leagle_training_diagnostics import packed_batch


def _run_steps(model, inputs, kwargs, *, amp, mode, saved=None):
    options, updates = resolve_ddp_options(
        SimpleNamespace(
            leagle_ddp_gradient_as_bucket_view=mode != "reference",
            leagle_ddp_batched_grad_copy=mode == "batched",
            leagle_ddp_bucket_cap_mb=1 if mode == "batched" else None,
        )
    )
    assert not updates
    wrapper = DistributedDataParallel(model, **options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if saved is not None:
        optimizer.load_state_dict(copy.deepcopy(saved))
    trainer = Trainer.__new__(Trainer)
    trainer.model, trainer.optimizers = wrapper, [optimizer]
    trainer.is_distributed = True
    trainer.config = SimpleNamespace(max_grad_norm=1.0)
    gradients, losses = [], []
    for step in range(3):
        trainer._optimizers_zero_grad()
        current = {name: value.clone() for name, value in inputs.items()}
        if step == 1 and dist.get_rank() == 1:
            current["loss_mask"].zero_()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            _, loss, metrics = wrapper(**current, **kwargs)
        loss, _, _ = trainer._prepare_supervised_loss(loss, metrics, current)
        (loss * 0.37).backward()
        gradients.append(
            {
                name: p.grad.clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
        )
        losses.append(loss.detach())
        trainer._clip_gradients()
        trainer._optimizers_step()
    outcome = (
        losses,
        gradients,
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )
    # Release the wrapper before creating the next reducer over a fresh model.
    del trainer, wrapper, optimizer
    return outcome


def _worker(rank, init_uri, batched_supported):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_uri,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=40),
    )
    try:
        torch.manual_seed(541)
        base = make_model(layers=1, loops=2, attention="sdpa")
        base.share_output_head()
        base.set_activation_checkpointing(True)
        torch.manual_seed(71 + rank)
        inputs = packed_batch()
        kwargs = {
            "ttt_steps": 3,
            "ttt_step_loss_decay": 0.8,
            "return_predictions": False,
            "loss_config": resolve_training_loss(
                '{"ce":0.1,"tv":0.9}', loss_implementation="eager"
            ),
            "leagle_chunked_loss": True,
            "leagle_loss_chunk_size": 2,
            "leagle_teacher_cache": True,
            "leagle_teacher_cache_supervised": True,
            "leagle_cache_teacher_indices": True,
            "leagle_cache_frozen_head_weights": True,
            "leagle_masked_head": True,
            "leagle_sparse_attention": True,
            "leagle_attn_chunk_size": 2,
            "leagle_reuse_training_kv": True,
        }
        for amp in (False, True):
            reference = _run_steps(
                copy.deepcopy(base), inputs, kwargs, amp=amp, mode="reference"
            )
            views = _run_steps(
                copy.deepcopy(base), inputs, kwargs, amp=amp, mode="views"
            )
            torch.testing.assert_close(views, reference, rtol=0, atol=0)
            if batched_supported:
                batched = _run_steps(
                    copy.deepcopy(base), inputs, kwargs, amp=amp, mode="batched"
                )
                # Batched copies may change float32 division/reduction rounding.
                torch.testing.assert_close(batched, reference, rtol=2e-5, atol=2e-7)
            # Switching DDP execution policy does not alter optimizer state loading.
            resumed = copy.deepcopy(base)
            resumed.load_state_dict(reference[2])
            resumed_reference = _run_steps(
                copy.deepcopy(resumed),
                inputs,
                kwargs,
                amp=amp,
                mode="reference",
                saved=reference[3],
            )
            resumed_views = _run_steps(
                resumed,
                inputs,
                kwargs,
                amp=amp,
                mode="views",
                saved=copy.deepcopy(reference[3]),
            )
            torch.testing.assert_close(resumed_views, resumed_reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo is unavailable")
def test_three_rank_ddp_updates_empty_rank_and_optimizer_resume(tmp_path):
    supported = (
        "batched_grad_copy" in inspect.signature(DistributedDataParallel).parameters
    )
    processes = mp.spawn(
        _worker, args=((tmp_path / "store").as_uri(), supported), nprocs=3, join=False
    )
    deadline = time.monotonic() + 100
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Three-rank DDP regression timed out")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
