from zspace.dataset import ToyModel
import pytest


def test_dataset_shapes():
    dataset = ToyModel()
    sample = dataset[0]

    assert sample[0].shape == (3, 32, 32)  # xa shape
    assert sample[1].shape == (10,)        # xb shape
    assert sample[2].shape == (10, 50)     # xc shape
