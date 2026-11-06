import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import streamlit_app


def _encoded_image(format_name="PNG", size=(24, 20)):
    pixels = np.random.default_rng(7).integers(
        0, 256, (size[1], size[0], 3), dtype=np.uint8
    )
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format=format_name)
    return buffer.getvalue()


def _config():
    return {
        "traditional_filters": {
            "gaussian": {"kernel_size": 5, "sigma": 1.0},
            "median": {"kernel_size": 3},
            "bilateral": {
                "diameter": 7,
                "sigma_color": 50,
                "sigma_space": 50,
            },
        }
    }


@pytest.mark.parametrize(
    ("name", "format_name"),
    [("first.png", "PNG"), ("second.jpg", "JPEG"), ("third.jpeg", "JPEG")],
)
def test_decode_multiple_upload_formats(name, format_name):
    image = streamlit_app.decode_uploaded_image(
        name, _encoded_image(format_name)
    )

    assert image.shape == (20, 24, 3)
    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        ("bad.gif", _encoded_image()),
        ("broken.png", b"not an image"),
        ("wrong-extension.jpg", _encoded_image("PNG")),
    ],
)
def test_decode_rejects_invalid_or_corrupted_uploads(name, contents):
    with pytest.raises(ValueError):
        streamlit_app.decode_uploaded_image(name, contents)


def test_decode_rejects_images_too_small_for_ssim():
    with pytest.raises(ValueError, match="at least"):
        streamlit_app.decode_uploaded_image(
            "small.png", _encoded_image(size=(6, 8))
        )


def test_batch_methods_are_isolated_and_share_identical_noisy_input(monkeypatch):
    calls = {"Gaussian": [], "Median": [], "Bilateral": []}
    latency_calls = []

    def fake_filter(name):
        def apply(image, **kwargs):
            calls[name].append((image, kwargs))
            return image.copy()
        return apply

    def fake_benchmark(func, num_runs, warmup):
        latency_calls.append(func)
        func()
        return {"mean_ms": 2.5}

    monkeypatch.setattr(
        streamlit_app, "apply_gaussian_filter", fake_filter("Gaussian")
    )
    monkeypatch.setattr(
        streamlit_app, "apply_median_filter", fake_filter("Median")
    )
    monkeypatch.setattr(
        streamlit_app, "apply_bilateral_filter", fake_filter("Bilateral")
    )
    monkeypatch.setattr(
        streamlit_app,
        "cnn_denoise_pipeline",
        lambda model, image, mode, device: image.copy(),
    )
    monkeypatch.setattr(streamlit_app, "benchmark_latency", fake_benchmark)

    result = streamlit_app.process_batch_image(
        clean_rgb=np.full((20, 24, 3), 0.5, dtype=np.float32),
        image_name="example.png",
        mode="Colour",
        sigma=25,
        seed=42,
        methods=["Gaussian", "Median"],
        config=_config(),
        latency_runs=1,
    )

    assert set(result["methods"]) == {"Gaussian", "Median"}
    assert len(calls["Gaussian"]) == 2
    assert len(calls["Median"]) == 2
    assert calls["Bilateral"] == []
    assert calls["Gaussian"][0][1] == _config()["traditional_filters"]["gaussian"]
    assert calls["Median"][0][1] == _config()["traditional_filters"]["median"]
    assert all(
        image is result["noisy"]
        for entries in calls.values()
        for image, _ in entries
    )
    assert len(latency_calls) == 2
    assert result["methods"]["Gaussian"]["latency_ms"] == 2.5


def test_batch_noise_is_reproducible_and_metrics_are_computed():
    clean = np.random.default_rng(3).random((20, 24, 3), dtype=np.float32)
    first = streamlit_app.process_batch_image(
        clean, "sample.png", "Colour", 15, 123, ["Gaussian"],
        _config(), latency_runs=1,
    )
    repeated = streamlit_app.process_batch_image(
        clean, "sample.png", "Colour", 15, 123, ["Gaussian"],
        _config(), latency_runs=1,
    )
    other_seed = streamlit_app.process_batch_image(
        clean, "sample.png", "Colour", 15, 124, ["Gaussian"],
        _config(), latency_runs=1,
    )

    assert np.array_equal(first["noisy"], repeated["noisy"])
    assert not np.array_equal(first["noisy"], other_seed["noisy"])
    metrics = first["methods"]["Gaussian"]
    assert metrics["psnr"] == streamlit_app.calculate_psnr(
        first["clean"], metrics["image"]
    )
    assert metrics["ssim"] == streamlit_app.calculate_ssim(
        first["clean"], metrics["image"]
    )
    assert first["clean"].shape == first["noisy"].shape == metrics["image"].shape


@pytest.mark.parametrize(
    ("mode", "expected_dimensions"),
    [("Colour", 3), ("Grayscale", 2)],
)
def test_batch_color_and_grayscale_processing(mode, expected_dimensions, monkeypatch):
    monkeypatch.setattr(
        streamlit_app,
        "cnn_denoise_pipeline",
        lambda model, image, selected_mode, device: image.copy(),
    )
    rgb_image = np.random.default_rng(4).random((20, 24, 3), dtype=np.float32)

    result = streamlit_app.process_batch_image(
        rgb_image,
        "mode.png",
        mode,
        25,
        42,
        ["CNN Autoencoder"],
        _config(),
        model=object(),
        latency_runs=1,
    )

    assert result["clean"].ndim == expected_dimensions
    assert result["noisy"].shape == result["clean"].shape
    assert result["methods"]["CNN Autoencoder"]["image"].shape == result["clean"].shape


def test_csv_and_zip_downloads_include_all_selected_method_results():
    results = [
        {
            "image_name": "photo.png",
            "noise_sigma": 25,
            "methods": {
                "Gaussian": {
                    "psnr": 30.1, "ssim": 0.91, "latency_ms": 1.2,
                    "image_png": b"gaussian png",
                },
                "CNN Autoencoder": {
                    "psnr": 31.2, "ssim": 0.93, "latency_ms": 2.3,
                    "image_png": b"cnn png",
                },
            },
        }
    ]

    csv_contents = streamlit_app.batch_results_to_csv(results).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(csv_contents)))
    assert list(rows[0]) == [
        "image_name", "method", "noise_sigma", "psnr", "ssim", "latency_ms"
    ]
    assert [row["method"] for row in rows] == ["Gaussian", "CNN Autoencoder"]

    with zipfile.ZipFile(io.BytesIO(streamlit_app.batch_results_to_zip(results))) as archive:
        assert archive.namelist() == [
            "001_photo/gaussian.png",
            "001_photo/cnn_autoencoder.png",
        ]
        assert archive.read("001_photo/gaussian.png") == b"gaussian png"
        assert archive.read("001_photo/cnn_autoencoder.png") == b"cnn png"
