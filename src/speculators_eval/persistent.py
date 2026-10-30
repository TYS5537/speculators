"""Opt-in dataset barriers for persistent, device-isolated evaluation workers.

The parent imports no model runtime. Each stdin job has an exact standalone argv
for replay and a unique completion marker published only after output validation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from tempfile import TemporaryDirectory


def _close_input(process):
    with suppress(OSError):
        process.stdin.close()


class PersistentWorkers:
    def __init__(self):
        self.stack = ExitStack()
        self.processes = {}
        self.pending = {}
        self.sequence = 0

    def __enter__(self):
        self.directory = Path(
            self.stack.enter_context(
                TemporaryDirectory(prefix="speculators-eval-workers-")
            )
        )
        return self

    def submit(self, index, command, env):
        from speculators_eval.parallel import stop_eval_worker  # noqa: PLC0415

        if index not in self.processes:
            process = subprocess.Popen(  # noqa: S603
                [*command, "--worker-job-stream"],
                env=env,
                stdin=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            self.processes[index] = process
            self.stack.callback(stop_eval_worker, process)
            self.stack.callback(_close_input, process)
        process = self.processes[index]
        if process.poll() is not None:
            raise RuntimeError(f"Persistent worker {index} exited before dispatch")
        self.sequence += 1
        completion = self.directory / f"{self.sequence}-{index}.done"
        self.pending[index] = completion
        job = {"argv": command[1:], "completion": str(completion)}
        process.stdin.write(json.dumps(job) + "\n")
        process.stdin.flush()
        return process

    def wait(self, processes, dataset, *, progress=None):
        while True:
            statuses = []
            for index, _, process in processes:
                code = process.poll()
                if code is not None:
                    raise RuntimeError(
                        f"{dataset} persistent worker {index} exited: {code}"
                    )
                statuses.append((index, 0 if self.pending[index].is_file() else None))
            if progress is not None:
                progress.update(statuses)
            if all(code == 0 for _, code in statuses):
                for index, _ in statuses:
                    self.pending.pop(index).unlink()
                return
            time.sleep(0.25)

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                for process in self.processes.values():
                    _close_input(process)
                for index, process in self.processes.items():
                    # ExitStack terminates/reaps children that miss the deadline.
                    with suppress(subprocess.TimeoutExpired):
                        code = process.wait(timeout=5)
                        if code != 0:
                            raise RuntimeError(
                                f"Persistent worker {index} failed during shutdown: "
                                f"{code}"
                            )
        finally:
            self.stack.__exit__(kind, value, traceback)


def capture_rng(device):
    """Capture after loading, matching a fresh worker's first evaluation sample."""
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device = torch.device(device)
    cpu = torch.get_rng_state()
    python = random.getstate()
    numpy = np.random.get_state()  # noqa: NPY002 - restore the existing global RNG
    backend = getattr(torch, device.type) if device.type != "cpu" else None
    accelerator = backend.get_rng_state(device) if backend is not None else None

    def restore():
        torch.set_rng_state(cpu)
        random.setstate(python)
        np.random.set_state(numpy)  # noqa: NPY002
        if backend is not None:
            backend.set_rng_state(accelerator, device)

    return restore


def validate_worker_models(target, draft):
    from speculators.models.utils import validate_kv_reuse_rope  # noqa: PLC0415

    # Dynamic/long RoPE can retain frequencies selected by a previous dataset.
    # Reusing a loaded model must not change that dataset's positional encoding.
    for rotary in (target.model.rotary_emb, draft.rotary_emb):
        validate_kv_reuse_rope(
            rotary, hint="disable --persistent-workers for dynamic RoPE"
        )


def serve_jobs(initial, runner, stop_ids):
    from speculators_eval import leagle, provenance, reporting  # noqa: PLC0415

    restore_rng = capture_rng(runner.device)
    mutable = {
        "datasets_root",
        "output_dir",
        "worker_progress_path",
        "worker_job_stream",
        "checkpoint_manifest",
        "checkpoint_identity",
    }
    for line in sys.stdin:
        job = json.loads(line)
        args = leagle.build_parser().parse_args(job["argv"][1:])
        leagle.validate_args(args)
        provenance.prepare_checkpoints(args)
        if args.checkpoint_identity != initial.checkpoint_identity or any(
            getattr(args, name) != value
            for name, value in vars(initial).items()
            if name not in mutable
        ):
            raise ValueError(
                "Persistent jobs must keep the same model and sampling settings"
            )
        leagle.save_provenance(
            args, runner.draft_model.config, argv=job["argv"], persistent=True
        )
        restore_rng()
        runner.args = args
        row, samples = leagle.evaluate_dataset(
            args.datasets_root, runner, args, stop_ids
        )
        provenance.validate_checkpoint_files(args)
        reporting.write_outputs(
            args.output_dir,
            [row],
            {} if args.skip_artifacts else {row["dataset"]: samples},
        )
        # The parent never reads partial results, or mistakes old output for this job.
        completion = Path(job["completion"])
        temporary = completion.with_suffix(".tmp")
        temporary.write_text("done\n", encoding="ascii")
        temporary.replace(completion)
