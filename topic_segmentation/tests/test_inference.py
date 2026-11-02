"""Sliding windows must score every unit of an article."""
import random

import pytest

from topic_segmentation.inference import windows

WINDOW = 1024


def unit_starts(lengths):
    starts, position = [], 0
    for length in lengths:
        starts.append(position)
        position += length
    return starts, position


def scored_units(starts, total, stride, protect):
    scored = set()
    for _, _, units in windows(starts, total, WINDOW, stride, protect):
        assert all(1 <= position < WINDOW for _, position in units)
        scored.update(unit for unit, _ in units)
    return scored


@pytest.mark.parametrize("protect", [False, True])
@pytest.mark.parametrize("stride", [256, 512, 768, 1024])
def test_random_articles(stride, protect):
    rng = random.Random(0)
    for _ in range(300):
        starts, total = unit_starts([rng.randint(1, 400) for _ in range(rng.randint(1, 40))])
        assert scored_units(starts, total, stride, protect) == set(range(len(starts)))


@pytest.mark.parametrize("protect", [False, True])
@pytest.mark.parametrize("stride", [512, 1024])
def test_unit_starting_at_the_last_window_position(stride, protect):
    # The window holds CLS plus 1023 tokens, so a unit starting at token 1023 belongs to the next window.
    starts, total = [0, 1023, 1500], 2000
    assert scored_units(starts, total, stride, protect) == {0, 1, 2}


@pytest.mark.parametrize("protect", [False, True])
def test_oversized_unit(protect):
    starts, total = [0, 10, 3000], 3100
    assert scored_units(starts, total, 512, protect) == {0, 1, 2}
