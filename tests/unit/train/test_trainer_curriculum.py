"""Tests for trainer-owned loss curriculum schedules."""

import pytest

from speculators.train.trainer import linear_base_curriculum_weight


def test_linear_base_curriculum_endpoints():
    assert linear_base_curriculum_weight(0, 5) == 1.0
    assert linear_base_curriculum_weight(2, 5) == 0.5
    assert linear_base_curriculum_weight(4, 5) == 0.0
    assert linear_base_curriculum_weight(20, 5) == 0.0


def test_linear_base_curriculum_single_step():
    assert linear_base_curriculum_weight(0, 1) == 1.0
    assert linear_base_curriculum_weight(1, 1) == 0.0


def test_linear_base_curriculum_rejects_invalid_total():
    with pytest.raises(ValueError):
        linear_base_curriculum_weight(0, 0)
