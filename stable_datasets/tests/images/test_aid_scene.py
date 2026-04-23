import shutil
import warnings

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.aid_scene import AIDScene


def test_aid_scene_dataset():
    if shutil.which("kaggle") is None:
        warnings.warn(
            "Kaggle CLI not found. Install it with `pip install kaggle` and configure ~/.kaggle/kaggle.json.",
            stacklevel=2,
        )
        pytest.skip("Kaggle CLI not found. Install `kaggle` and configure authentication to run this test.")

    try:
        aid_all = AIDScene(split="all")
    except RuntimeError as error:
        if "Kaggle download failed" in str(error):
            pytest.fail(
                "Kaggle download failed. Please configure Kaggle API credentials at "
                "~/.kaggle/kaggle.json (with 'username' and 'key') and run "
                "`chmod 600 ~/.kaggle/kaggle.json`."
            )
        raise
    assert len(aid_all) > 0, "Expected non-empty dataset."

    sample = aid_all[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert len(image_np.shape) == 3, f"Image should have 3 dimensions (H, W, C), got shape {image_np.shape}"
    assert image_np.shape[2] == 3, f"Image should have 3 channels (RGB), got {image_np.shape[2]} channels"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 30, f"Label should be between 0 and 29 (30 classes), got {label}."

    expected_labels = [
        "Airport",
        "BareLand",
        "BaseballField",
        "Beach",
        "Bridge",
        "Center",
        "Church",
        "Commercial",
        "DenseResidential",
        "Desert",
        "Farmland",
        "Forest",
        "Industrial",
        "Meadow",
        "MediumResidential",
        "Mountain",
        "Park",
        "Parking",
        "Playground",
        "Pond",
        "Port",
        "RailwayStation",
        "Resort",
        "River",
        "School",
        "SparseResidential",
        "Square",
        "Stadium",
        "StorageTanks",
        "Viaduct",
    ]
    assert aid_all.info.features["label"].names == expected_labels
