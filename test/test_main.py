import os
import csv
import numpy as np
import cv2
import pytest
from main import get_image_info, scan_folder, write_report


@pytest.fixture
def sample_image(tmp_path):
    """Creates a real temporary image file for testing."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    path = str(tmp_path / "test.jpg")
    cv2.imwrite(path, img)
    return path


def test_get_image_info_shape(sample_image):
    info = get_image_info(sample_image)
    assert info["width"] == 200
    assert info["height"] == 100
    assert info["channels"] == 3


def test_get_image_info_fields(sample_image):
    info = get_image_info(sample_image)
    assert "filename" in info
    assert "size_kb" in info
    assert info["filename"] == "test.jpg"


def test_get_image_info_invalid_path():
    with pytest.raises(ValueError):
        get_image_info("nonexistent.jpg")


def test_scan_folder(tmp_path):
    # Create two fake images
    for name in ["a.jpg", "b.png"]:
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / name), img)

    results = scan_folder(str(tmp_path))
    assert len(results) == 2
    assert results[0]["filename"] == "a.jpg"


def test_scan_folder_ignores_non_images(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    results = scan_folder(str(tmp_path))
    assert len(results) == 0


def test_write_report(tmp_path):
    data = [{"filename": "a.jpg", "width": 100, "height": 100, "channels": 3, "size_kb": 5.0}]
    output = str(tmp_path / "report.csv")
    write_report(data, output)

    assert os.path.exists(output)
    with open(output) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["filename"] == "a.jpg"
