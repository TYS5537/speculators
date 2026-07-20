from unittest.mock import MagicMock, patch

import pytest

from speculators.train.trainer import _StepTimer


def test_disabled_timer_returns_none():
    timer = _StepTimer(enabled=False)
    timer.mark_value("start", 0.0)
    timer.mark("fetch")
    timer.mark("fwd")
    timer.mark("bwd")
    timer.mark("opt")
    assert timer.now() is None
    assert timer.profile(num_tokens=1024) is None


def test_enabled_timer_returns_profile():
    timer = _StepTimer(enabled=True)
    mock_sync = MagicMock()

    with (
        patch("speculators.train.trainer._synchronize_device", mock_sync),
        patch(
            "speculators.train.trainer.time.perf_counter",
            side_effect=[0.1, 0.3, 0.5, 0.6, 0.6],
        ),
    ):
        timer.mark_value("start", 0.0)
        timer.mark("fetch")
        timer.mark("fwd")
        timer.mark("bwd")
        timer.mark("opt")
        t_next = timer.now()

    assert mock_sync.call_count == 5
    assert t_next == 0.6

    profile = timer.profile(num_tokens=4096)
    assert profile is not None
    assert profile["fetch_ms"] == (0.1 - 0.0) * 1000
    assert profile["fwd_ms"] == (0.3 - 0.1) * 1000
    assert profile["bwd_ms"] == (0.5 - 0.3) * 1000
    assert profile["opt_ms"] == (0.6 - 0.5) * 1000
    assert profile["step_ms"] == (0.6 - 0.0) * 1000
    assert profile["tokens_per_s"] == 4096 / 0.6
    assert profile["fetch_frac"] == 100 / 600


def test_profile_returns_none_when_marks_incomplete():
    timer = _StepTimer(enabled=True)
    timer.mark_value("start", 0.0)
    timer.mark("fetch")
    assert timer.profile(num_tokens=1024) is None


def test_mark_records_even_if_synchronize_fails():
    timer = _StepTimer(enabled=True)
    with (
        patch(
            "speculators.train.trainer._synchronize_device",
            side_effect=RuntimeError("sync failed"),
        ),
        patch("speculators.train.trainer.time.perf_counter", return_value=1.5),
        pytest.raises(RuntimeError, match="sync failed"),
    ):
        timer.mark("fetch")
    assert timer._marks["fetch"] == 1.5


def test_disabled_to_enabled_transition():
    """Simulate log_freq=2: a disabled step followed by an enabled step.

    The disabled step's ``timer.now()`` returns None so the training loop
    falls back to ``time.perf_counter()`` for ``t_before_fetch``.  The
    subsequent enabled step feeds that value into ``mark_value("start", ...)``.
    Verify the profile is valid with a realistic ``start`` mark.
    """
    timer = _StepTimer()

    # --- disabled step (global_step=1, log_freq=2 → 1%2 != 0) ---
    timer.reset(enabled=False)
    timer.mark_value("start", 1.0)
    timer.mark("fetch")
    timer.mark("fwd")
    timer.mark("bwd")
    timer.mark("opt")
    assert timer.now() is None
    assert timer.profile(num_tokens=512) is None

    # Simulate the fallback: t_before_fetch = timer.now() or time.perf_counter()
    t_before_fetch = 2.0  # stands in for the perf_counter() fallback

    # --- enabled step (global_step=2, log_freq=2 → 2%2 == 0) ---
    timer.reset(enabled=True)
    timer.mark_value("start", t_before_fetch)

    with (
        patch("speculators.train.trainer._synchronize_device"),
        patch(
            "speculators.train.trainer.time.perf_counter",
            side_effect=[2.1, 2.4, 2.5, 2.6, 2.6],
        ),
    ):
        timer.mark("fetch")
        timer.mark("fwd")
        timer.mark("bwd")
        timer.mark("opt")
        t_next = timer.now()

    assert t_next == 2.6
    profile = timer.profile(num_tokens=2048)
    assert profile is not None
    assert profile["fetch_ms"] == (2.1 - 2.0) * 1000
    assert profile["fwd_ms"] == (2.4 - 2.1) * 1000
    assert profile["bwd_ms"] == (2.5 - 2.4) * 1000
    assert profile["opt_ms"] == (2.6 - 2.5) * 1000
    assert profile["step_ms"] == (2.6 - 2.0) * 1000
    assert profile["tokens_per_s"] == pytest.approx(2048 / 0.6)


def test_zero_step_ms_returns_zero_throughput():
    timer = _StepTimer(enabled=True)
    timer.mark_value("start", 1.0)
    with (
        patch("speculators.train.trainer._synchronize_device"),
        patch("speculators.train.trainer.time.perf_counter", return_value=1.0),
    ):
        timer.mark("fetch")
        timer.mark("fwd")
        timer.mark("bwd")
        timer.mark("opt")
    profile = timer.profile(num_tokens=4096)
    assert profile is not None
    assert profile["tokens_per_s"] == 0.0
    assert profile["fetch_frac"] == 0.0
