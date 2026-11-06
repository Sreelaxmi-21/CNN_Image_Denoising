"""
Streamlit interface for the CNN Image Denoising project.

Run with:
    streamlit run app/streamlit_app.py
"""

import csv
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Set up project imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluate import cnn_denoise_image, load_cnn_model
from src.metrics import benchmark_latency, calculate_psnr, calculate_ssim
from src.noise import add_gaussian_noise_numpy
from src.traditional import (
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_median_filter,
)

st.set_page_config(
    page_title="CNN Image Denoising Demo",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CHECKPOINT_PATH = str(
    _PROJECT_ROOT
    / "results"
    / "experiments"
    / "stride16_epochs30"
    / "checkpoints"
    / "best_model.pth"
)
_CONFIG_PATH = str(_PROJECT_ROOT / "configs" / "config.yaml")
_DEFAULT_IMAGES = sorted(
    (_PROJECT_ROOT / "data" / "test_benchmark").glob("*.png")
)

SIGMA_OPTIONS = [15, 25, 50]
_MAX_IMAGE_PIXELS = 40_000_000
_MIN_IMAGE_SIDE = 7
_BATCH_METHODS = ("CNN Autoencoder", "Gaussian", "Median", "Bilateral")
_BATCH_CSV_FIELDS = (
    "image_name",
    "method",
    "noise_sigma",
    "psnr",
    "ssim",
    "latency_ms",
)


@st.cache_resource(show_spinner="Loading CNN model...")
def _load_model(checkpoint_path: str = _CHECKPOINT_PATH):
    """Load and cache the trained CNN checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None
    return load_cnn_model(checkpoint_path, device="cpu")


@st.cache_data
def _load_config():
    import yaml

    with open(_CONFIG_PATH, "r") as config_file:
        return yaml.safe_load(config_file)


def pil_to_float32(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to RGB float32 pixels in [0, 1]."""
    return np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0


def float32_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert normalized grayscale or RGB pixels to a PIL image."""
    arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr_u8.ndim == 2:
        return Image.fromarray(arr_u8, mode="L")
    return Image.fromarray(arr_u8, mode="RGB")


def timed_denoise(func, *args, **kwargs):
    """Run a function and return its result and elapsed milliseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def cnn_denoise_pipeline(
    model,
    noisy_np: np.ndarray,
    mode: str,
    device: str = "cpu",
) -> np.ndarray:
    """Run CNN inference in colour or grayscale mode."""
    if mode == "Colour":
        ycrcb = cv2.cvtColor(noisy_np, cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb[:, :, 0]
        cr_channel = ycrcb[:, :, 1]
        cb_channel = ycrcb[:, :, 2]

        y_3ch = np.stack([y_channel, y_channel, y_channel], axis=-1)
        y_den_3ch = cnn_denoise_image(model, y_3ch, device=device)
        y_den = np.clip(
            np.mean(y_den_3ch, axis=-1), 0.0, 1.0
        ).astype(np.float32)

        ycrcb_recon = np.stack([y_den, cr_channel, cb_channel], axis=-1)
        rgb_den = cv2.cvtColor(ycrcb_recon, cv2.COLOR_YCrCb2RGB)
        return np.clip(rgb_den, 0.0, 1.0).astype(np.float32)

    gray_3ch = np.stack([noisy_np, noisy_np, noisy_np], axis=-1)
    cnn_out_3ch = cnn_denoise_image(model, gray_3ch, device=device)
    return np.clip(
        np.mean(cnn_out_3ch, axis=-1), 0.0, 1.0
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Batch processing helpers
# ---------------------------------------------------------------------------


def decode_uploaded_image(file_name: str, image_bytes: bytes) -> np.ndarray:
    """Validate an uploaded PNG/JPEG and return RGB float32 pixels."""
    extension = Path(file_name).suffix.lower()
    expected_format = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
    }.get(extension)

    if expected_format is None:
        raise ValueError("Only PNG, JPG, and JPEG images are supported.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != expected_format:
                raise ValueError(
                    f"File extension does not match image format "
                    f"({image.format})."
                )

            width, height = image.size
            if width < _MIN_IMAGE_SIDE or height < _MIN_IMAGE_SIDE:
                raise ValueError(
                    f"Image must be at least {_MIN_IMAGE_SIDE}x"
                    f"{_MIN_IMAGE_SIDE} pixels."
                )
            if width * height > _MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image exceeds the {_MAX_IMAGE_PIXELS:,}-pixel limit."
                )
            image.verify()

        with Image.open(io.BytesIO(image_bytes)) as image:
            return np.asarray(
                image.convert("RGB"), dtype=np.float32
            ) / 255.0

    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Image could not be decoded: {exc}") from exc


def prepare_batch_image(clean_rgb: np.ndarray, mode: str) -> np.ndarray:
    """Convert RGB pixels to colour or grayscale mode."""
    if mode == "Colour":
        return clean_rgb.astype(np.float32, copy=True)
    if mode == "Grayscale":
        return cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2GRAY).astype(
            np.float32
        )
    raise ValueError("Mode must be 'Colour' or 'Grayscale'.")


def configured_checkpoint_path(config: dict) -> str:
    """Resolve the configured checkpoint relative to the project root."""
    checkpoint = (
        config.get("evaluation", {}).get("inference_checkpoint")
        or os.path.join(
            config.get("training", {}).get(
                "checkpoint_dir", "results/checkpoints"
            ),
            "best_model.pth",
        )
    )
    path = Path(checkpoint)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def process_batch_image(
    clean_rgb: np.ndarray,
    image_name: str,
    mode: str,
    sigma: int,
    seed: int,
    methods,
    config: dict,
    model=None,
    latency_runs: int = 3,
) -> dict:
    """Run selected methods on one shared noisy input and compute metrics."""
    if sigma not in SIGMA_OPTIONS:
        raise ValueError(f"Noise sigma must be one of {SIGMA_OPTIONS}.")

    selected_methods = list(methods)
    if not selected_methods or any(
        method not in _BATCH_METHODS for method in selected_methods
    ):
        raise ValueError("Select one or more supported denoising methods.")

    if "CNN Autoencoder" in selected_methods and model is None:
        raise ValueError(
            "CNN Autoencoder was selected, but the model is unavailable."
        )
    if latency_runs < 1:
        raise ValueError("latency_runs must be at least 1.")

    clean = prepare_batch_image(clean_rgb, mode)
    noisy = add_gaussian_noise_numpy(clean, sigma=sigma, seed=seed)
    filter_config = config.get("traditional_filters", {})

    runners = {
        "CNN Autoencoder": lambda: cnn_denoise_pipeline(
            model, noisy, mode, device="cpu"
        ),
        "Gaussian": lambda: apply_gaussian_filter(
            noisy, **filter_config.get("gaussian", {})
        ),
        "Median": lambda: apply_median_filter(
            noisy, **filter_config.get("median", {})
        ),
        "Bilateral": lambda: apply_bilateral_filter(
            noisy, **filter_config.get("bilateral", {})
        ),
    }

    outputs = {}
    for method in selected_methods:
        runner = runners[method]
        denoised = runner()
        latency_ms = benchmark_latency(
            runner, num_runs=latency_runs, warmup=2
        )["mean_ms"]

        if denoised.shape != clean.shape:
            raise ValueError(
                f"{method} returned shape {denoised.shape}; "
                f"expected {clean.shape}."
            )

        outputs[method] = {
            "image": denoised,
            "psnr": calculate_psnr(clean, denoised),
            "ssim": calculate_ssim(clean, denoised),
            "latency_ms": latency_ms,
        }

    return {
        "image_name": image_name,
        "mode": mode,
        "noise_sigma": sigma,
        "clean": clean,
        "noisy": noisy,
        "methods": outputs,
    }


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode normalized grayscale or RGB pixels as PNG bytes."""
    buffer = io.BytesIO()
    float32_to_pil(image).save(buffer, format="PNG")
    return buffer.getvalue()


def batch_results_to_csv(results) -> bytes:
    """Serialize batch metrics as CSV."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_BATCH_CSV_FIELDS)
    writer.writeheader()

    for result in results:
        for method, metrics in result["methods"].items():
            writer.writerow(
                {
                    "image_name": result["image_name"],
                    "method": method,
                    "noise_sigma": result["noise_sigma"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "latency_ms": metrics["latency_ms"],
                }
            )

    return buffer.getvalue().encode("utf-8")


def batch_results_to_zip(results) -> bytes:
    """Bundle denoised PNGs into a ZIP archive."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for index, result in enumerate(results, start=1):
            image_stem = Path(result["image_name"]).stem

            for method, metrics in result["methods"].items():
                safe_method = method.lower().replace(" ", "_")
                file_path = f"{index:03d}_{image_stem}/{safe_method}.png"
                archive.writestr(file_path, metrics["image_png"])

    return buffer.getvalue()


def _render_batch_image_studio():
    """Render the batch image processing workflow."""
    st.title("Batch Image Processing Studio")
    st.caption(
        "Uploaded images are treated as clean references, then synthetic "
        "AWGN is added. Metrics do not represent real-world noisy image "
        "performance."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PNG, JPG, or JPEG images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help=(
            "Images must be at least 7x7 pixels and no larger than "
            f"{_MAX_IMAGE_PIXELS:,} pixels."
        ),
        key="batch_image_uploader",
    )
    mode = st.selectbox(
        "Processing mode", ["Colour", "Grayscale"], key="batch_mode"
    )
    sigma = st.selectbox(
        "Synthetic AWGN sigma",
        SIGMA_OPTIONS,
        key="batch_sigma",
    )
    seed = st.number_input(
        "Noise seed",
        min_value=0,
        value=42,
        step=1,
        key="batch_noise_seed",
    )
    methods = st.multiselect(
        "Denoising methods",
        _BATCH_METHODS,
        default=list(_BATCH_METHODS),
        key="batch_methods",
    )

    if not uploaded_files:
        st.info("Upload images to begin batch processing.")
        return
    if not methods:
        st.warning("Select at least one denoising method.")
        return

    if st.button("Process batch", type="primary", key="process_batch"):
        config = _load_config()
        model = None

        if "CNN Autoencoder" in methods:
            checkpoint_path = configured_checkpoint_path(config)
            if not os.path.isfile(checkpoint_path):
                st.error(
                    f"Configured CNN checkpoint not found: {checkpoint_path}"
                )
                return

            try:
                model = _load_model(checkpoint_path)
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f"Could not load the configured CNN checkpoint: {exc}")
                return

            if model is None:
                st.error(
                    f"Configured CNN checkpoint not found: {checkpoint_path}"
                )
                return

        processed = []
        progress = st.progress(0.0, text="Preparing batch...")

        for index, uploaded in enumerate(uploaded_files, start=1):
            try:
                clean_rgb = decode_uploaded_image(
                    uploaded.name, uploaded.getvalue()
                )
                result = process_batch_image(
                    clean_rgb=clean_rgb,
                    image_name=uploaded.name,
                    mode=mode,
                    sigma=int(sigma),
                    seed=int(seed),
                    methods=methods,
                    config=config,
                    model=model,
                )

                result["clean_png"] = image_to_png_bytes(
                    result.pop("clean")
                )
                result["noisy_png"] = image_to_png_bytes(
                    result.pop("noisy")
                )

                for metrics in result["methods"].values():
                    metrics["image_png"] = image_to_png_bytes(
                        metrics.pop("image")
                    )

                processed.append(result)

            except (ValueError, OSError, RuntimeError) as exc:
                st.warning(f"Skipped {uploaded.name}: {exc}")

            progress.progress(
                index / len(uploaded_files),
                text=f"Processed {index} of {len(uploaded_files)} uploads",
            )

        st.session_state["batch_results"] = processed

    results = st.session_state.get("batch_results", [])
    if not results:
        return

    st.subheader("Batch summary")
    import pandas as pd

    summary_rows = [
        {
            "Image": result["image_name"],
            "Method": method,
            "PSNR (dB)": metrics["psnr"],
            "SSIM": metrics["ssim"],
            "Latency (ms)": metrics["latency_ms"],
        }
        for result in results
        for method, metrics in result["methods"].items()
    ]

    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.dataframe(
        summary.groupby("Method", as_index=False)[
            ["PSNR (dB)", "SSIM", "Latency (ms)"]
        ].mean(),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Results gallery")
    st.caption(
        "Choose an uploaded image to compare its clean original, noisy input, "
        "and denoised outputs."
    )

    image_names = [result["image_name"] for result in results]
    selected_name = st.selectbox(
        "Image to compare",
        image_names,
        format_func=lambda name: (
            f"{name} · {image_names.index(name) + 1} of {len(image_names)}"
        ),
        key="batch_gallery_image",
    )
    image_index = image_names.index(selected_name)
    result = results[image_index]

    st.caption(
        f"{result['mode']} mode · synthetic AWGN "
        f"sigma={result['noise_sigma']} · seed={seed}"
    )

    columns = st.columns(2 + len(result["methods"]))
    columns[0].image(
        result["clean_png"],
        caption="Clean original",
        use_container_width=True,
    )
    columns[1].image(
        result["noisy_png"],
        caption="Noisy input",
        use_container_width=True,
    )

    for method_index, (method, metrics) in enumerate(
        result["methods"].items(), start=2
    ):
        columns[method_index].image(
            metrics["image_png"],
            caption=(
                f"{method} · PSNR {metrics['psnr']:.2f} dB · "
                f"SSIM {metrics['ssim']:.4f} · "
                f"{metrics['latency_ms']:.2f} ms"
            ),
            use_container_width=True,
        )
        st.download_button(
            f"Download {method} — {result['image_name']}",
            data=metrics["image_png"],
            file_name=(
                f"{Path(result['image_name']).stem}_"
                f"{method.lower().replace(' ', '_')}.png"
            ),
            mime="image/png",
            key=f"download_{image_index}_{method_index}",
        )

    st.subheader("Download centre")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download all denoised images (ZIP)",
        data=batch_results_to_zip(results),
        file_name="batch_denoised_images.zip",
        mime="application/zip",
    )
    download_columns[1].download_button(
        "Download metrics (CSV)",
        data=batch_results_to_csv(results),
        file_name="batch_denoising_metrics.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Research analytics dashboard
# ---------------------------------------------------------------------------
_BENCHMARK_CSV_CANDIDATES = (
    _PROJECT_ROOT
    / "results"
    / "experiments"
    / "stride16_epochs30"
    / "tables"
    / "evaluation_results.csv",
    _PROJECT_ROOT / "results" / "tables" / "evaluation_results.csv",
)
_BENCHMARK_COLUMNS = {
    "image",
    "sigma",
    "method",
    "psnr",
    "ssim",
    "latency_ms",
}


def load_benchmark_results(path=None):
    """Load and validate the per-image benchmark CSV."""
    import pandas as pd

    csv_path = Path(path) if path else next(
        (
            candidate
            for candidate in _BENCHMARK_CSV_CANDIDATES
            if candidate.is_file()
        ),
        None,
    )
    if csv_path is None:
        raise FileNotFoundError(
            "No benchmark CSV found in results/tables or the active "
            "experiment tables folder."
        )

    frame = pd.read_csv(csv_path)
    missing = _BENCHMARK_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "Benchmark CSV is missing required columns: "
            f"{', '.join(sorted(missing))}."
        )

    frame = frame[
        ["image", "sigma", "method", "psnr", "ssim", "latency_ms"]
    ].copy()
    frame["sigma"] = pd.to_numeric(frame["sigma"], errors="coerce")

    for metric in ("psnr", "ssim", "latency_ms"):
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    frame = frame.dropna(
        subset=["image", "sigma", "method", "psnr", "ssim", "latency_ms"]
    )
    frame["method"] = (
        frame["method"].astype(str).replace({"CNN": "CNN Autoencoder"})
    )
    frame["sigma"] = frame["sigma"].astype(int)

    if frame.empty:
        raise ValueError("Benchmark CSV contains no valid result rows.")

    return frame


def _render_research_analytics():
    """Render summaries and trends from saved Set12 benchmark results."""
    import pandas as pd

    st.title("Research Analytics Dashboard")
    st.caption(
        "Set12 benchmark results aggregated across images. These are saved "
        "evaluation results, not live inference. Latency depends on the "
        "hardware and run conditions used during evaluation."
    )

    try:
        results = load_benchmark_results()
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
        st.error(f"Could not load benchmark results: {exc}")
        return

    sigmas = sorted(results["sigma"].unique().tolist())
    selected_sigmas = st.multiselect(
        "Noise levels to include",
        sigmas,
        default=sigmas,
        key="analytics_sigmas",
    )

    if not selected_sigmas:
        st.info("Select at least one noise level to view the analytics.")
        return

    filtered = results[results["sigma"].isin(selected_sigmas)].copy()

    method_summary = filtered.groupby("method", as_index=False).agg(
        **{
            "Mean PSNR (dB)": ("psnr", "mean"),
            "Mean SSIM": ("ssim", "mean"),
            "Mean latency (ms)": ("latency_ms", "mean"),
        }
    )
    by_sigma = filtered.groupby(
        ["sigma", "method"], as_index=False
    ).agg(
        psnr=("psnr", "mean"),
        ssim=("ssim", "mean"),
        latency_ms=("latency_ms", "mean"),
    )

    best_psnr = method_summary.loc[
        method_summary["Mean PSNR (dB)"].idxmax()
    ]
    best_ssim = method_summary.loc[
        method_summary["Mean SSIM"].idxmax()
    ]
    fastest = method_summary.loc[
        method_summary["Mean latency (ms)"].idxmin()
    ]

    kpi_columns = st.columns(3)
    kpi_columns[0].metric(
        "Best mean PSNR",
        f"{best_psnr['Mean PSNR (dB)']:.2f} dB",
        best_psnr["method"],
    )
    kpi_columns[1].metric(
        "Best mean SSIM",
        f"{best_ssim['Mean SSIM']:.4f}",
        best_ssim["method"],
    )
    kpi_columns[2].metric(
        "Fastest mean latency",
        f"{fastest['Mean latency (ms)']:.2f} ms",
        fastest["method"],
    )

    st.caption(
        f"{filtered['image'].nunique()} benchmark images · "
        f"{len(selected_sigmas)} noise levels · "
        f"{len(filtered):,} image/method evaluations"
    )

    st.subheader("Performance across noise levels")
    st.caption(
        "Each line shows one denoising method. Higher PSNR and SSIM indicate "
        "better image quality."
    )

    for metric, title in (
        ("psnr", "Mean PSNR (dB) — higher is better"),
        ("ssim", "Mean SSIM — higher is better"),
    ):
        chart_data = (
            by_sigma.pivot(
                index="sigma",
                columns="method",
                values=metric,
            )
            .sort_index()
        )
        st.markdown(f"**{title}**")
        st.line_chart(chart_data, use_container_width=True)

    st.subheader("Compare methods at one noise level")
    selected_sigma = st.selectbox(
        "Noise sigma",
        selected_sigmas,
        index=len(selected_sigmas) - 1,
        key="analytics_detail_sigma",
    )
    detail = by_sigma[by_sigma["sigma"] == selected_sigma].set_index(
        "method"
    )

    import matplotlib.pyplot as plt

    for metric, title in (
        ("psnr", "Mean PSNR (dB) — higher is better"),
        ("ssim", "Mean SSIM — higher is better"),
        ("latency_ms", "Mean latency (ms) — lower is faster"),
    ):
        values = detail[metric].sort_values()
        dark_bg = "#0E1117"
        light_text = "#FAFAFA"

        figure, axis = plt.subplots(
            figsize=(9, 2.8),
            facecolor=dark_bg,
        )
        axis.set_facecolor(dark_bg)
        bars = axis.barh(
            values.index,
            values.values,
            color="#55A6E8",
        )
        axis.set_title(
            title,
            loc="left",
            fontsize=11,
            pad=10,
            color=light_text,
        )
        axis.set_xlabel(
            title.split(" — ")[0],
            color=light_text,
        )
        axis.grid(axis="x", color="#454955", alpha=0.65)
        axis.set_axisbelow(True)
        axis.tick_params(colors=light_text)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_color("#777B85")
        axis.spines["left"].set_color("#777B85")

        max_value = max(float(values.max()), 1.0)
        axis.set_xlim(0, max_value * 1.16)

        for bar, value in zip(bars, values.values):
            axis.text(
                value + max_value * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=9,
                color=light_text,
            )

        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    st.subheader("Benchmark summary")
    st.dataframe(method_summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered benchmark rows (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="set12_benchmark_filtered.csv",
        mime="text/csv",
        key="download_analytics_csv",
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main():
    workflow = st.sidebar.radio(
        "Workflow",
        [
            "Single Image Demo",
            "Batch Image Processing Studio",
            "Research Analytics",
        ],
        key="application_workflow",
    )

    if workflow == "Research Analytics":
        _render_research_analytics()
        return

    if workflow == "Batch Image Processing Studio":
        _render_batch_image_studio()
        return

    st.sidebar.title("CONTROLS")
    st.sidebar.markdown("---")

    source_option = st.sidebar.radio(
        "Image Source",
        ["Upload image", "Use Set12 benchmark image"],
        index=1,
    )

    clean_rgb = None
    image_label = ""

    if source_option == "Upload image":
        uploaded = st.sidebar.file_uploader(
            "Upload a PNG or JPG image",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            max_dim = 512
            width, height = pil_img.size

            if max(width, height) > max_dim:
                scale = max_dim / max(width, height)
                pil_img = pil_img.resize(
                    (int(width * scale), int(height * scale)),
                    Image.LANCZOS,
                )
                st.sidebar.info(
                    f"Resized to {pil_img.size[0]}x{pil_img.size[1]} "
                    "for CPU speed."
                )

            clean_rgb = pil_to_float32(pil_img)
            image_label = uploaded.name

    else:
        if _DEFAULT_IMAGES:
            image_choices = {path.name: path for path in _DEFAULT_IMAGES}
            selected_name = st.sidebar.selectbox(
                "Select benchmark image",
                list(image_choices.keys()),
            )
            pil_img = Image.open(image_choices[selected_name]).convert("RGB")
            clean_rgb = pil_to_float32(pil_img)
            image_label = selected_name
        else:
            st.sidebar.warning(
                "No Set12 benchmark images found. "
                "Run scripts/download_dataset.py first."
            )

    st.sidebar.markdown("---")

    mode_option = st.sidebar.radio(
        "Mode",
        ["Colour", "Grayscale"],
        index=0,
        help=(
            "Colour: CNN denoises luminance Y in YCrCb space and preserves "
            "chrominance Cr/Cb. Grayscale: single-channel processing."
        ),
    )

    st.sidebar.markdown("---")

    sigma = st.sidebar.select_slider(
        "AWGN Noise Level (sigma)",
        options=SIGMA_OPTIONS,
        value=25,
        help=(
            "Higher sigma means more noise. Standard levels: "
            "15 (mild), 25 (moderate), 50 (severe)."
        ),
    )

    noise_seed = st.sidebar.number_input(
        "Noise seed (for reproducibility)",
        value=42,
        min_value=0,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Methods evaluated:**")
    st.sidebar.markdown("- Gaussian Filter (k=5, sigma=1.0)")
    st.sidebar.markdown("- Median Filter (k=3)")
    st.sidebar.markdown("- Bilateral Filter (d=7, sigma_c=50, sigma_s=50)")
    st.sidebar.markdown("- CNN Autoencoder (426,435 params)")

    st.title("Deep Learning-Based Image Denoising")
    st.markdown(
        "**Research Demo** — Comparison of classical spatial filters and a "
        "CNN Autoencoder under Additive White Gaussian Noise (AWGN)."
    )
    st.markdown("---")

    if clean_rgb is None:
        st.info("Select or upload an image from the sidebar to begin.")
        return

    if mode_option == "Grayscale":
        clean_np = cv2.cvtColor(
            clean_rgb, cv2.COLOR_RGB2GRAY
        ).astype(np.float32)
    else:
        clean_np = clean_rgb.copy()

    model = _load_model()
    config = _load_config()

    if model is None:
        st.error(
            f"CNN checkpoint not found at:\n`{_CHECKPOINT_PATH}`\n\n"
            "Ensure the checkpoint exists at the path specified."
        )

    noisy_np = add_gaussian_noise_numpy(
        clean_np, sigma=sigma, seed=int(noise_seed)
    )

    filter_config = (
        config.get("traditional_filters", {}) if config else {}
    )
    gaussian_params = filter_config.get(
        "gaussian", {"kernel_size": 5, "sigma": 1.0}
    )
    median_params = filter_config.get("median", {"kernel_size": 3})
    bilateral_params = filter_config.get(
        "bilateral",
        {
            "diameter": 7,
            "sigma_color": 50.0,
            "sigma_space": 50.0,
        },
    )

    with st.spinner("Running denoising methods..."):
        gaussian_out, gaussian_ms = timed_denoise(
            apply_gaussian_filter, noisy_np, **gaussian_params
        )
        median_out, median_ms = timed_denoise(
            apply_median_filter, noisy_np, **median_params
        )
        bilateral_out, bilateral_ms = timed_denoise(
            apply_bilateral_filter, noisy_np, **bilateral_params
        )

        if model is not None:
            cnn_out, cnn_ms = timed_denoise(
                cnn_denoise_pipeline,
                model,
                noisy_np,
                mode_option,
                device="cpu",
            )
        else:
            cnn_out = np.zeros_like(clean_np)
            cnn_ms = 0.0

    st.subheader(
        f"Visual Comparison — {image_label} | {mode_option} "
        f"Mode | sigma = {sigma}"
    )

    row1_columns = st.columns(3)
    row2_columns = st.columns(3)

    def metric_caption(arr, label):
        psnr = calculate_psnr(clean_np, arr)
        ssim = calculate_ssim(clean_np, arr)
        psnr_text = (
            f"{psnr:.2f} dB"
            if psnr != float("inf")
            else "inf dB"
        )
        return (
            f"**{label}**  \nPSNR: `{psnr_text}`  \n"
            f"SSIM: `{ssim:.4f}`"
        )

    with row1_columns[0]:
        st.image(
            float32_to_pil(clean_np),
            caption=f"Original (Clean - {mode_option})",
            use_container_width=True,
        )

    with row1_columns[1]:
        noisy_psnr = calculate_psnr(clean_np, noisy_np)
        noisy_ssim = calculate_ssim(clean_np, noisy_np)
        noisy_psnr_text = (
            f"{noisy_psnr:.2f} dB"
            if noisy_psnr != float("inf")
            else "inf dB"
        )
        st.image(
            float32_to_pil(noisy_np),
            caption=(
                f"**Noisy ({mode_option}, sigma={sigma})**  \n"
                f"PSNR: `{noisy_psnr_text}`  \n"
                f"SSIM: `{noisy_ssim:.4f}`"
            ),
            use_container_width=True,
        )

    with row1_columns[2]:
        st.image(
            float32_to_pil(gaussian_out),
            caption=metric_caption(gaussian_out, "Gaussian Filter"),
            use_container_width=True,
        )

    with row2_columns[0]:
        st.image(
            float32_to_pil(median_out),
            caption=metric_caption(median_out, "Median Filter"),
            use_container_width=True,
        )

    with row2_columns[1]:
        st.image(
            float32_to_pil(bilateral_out),
            caption=metric_caption(bilateral_out, "Bilateral Filter"),
            use_container_width=True,
        )

    with row2_columns[2]:
        if model is not None:
            st.image(
                float32_to_pil(cnn_out),
                caption=metric_caption(cnn_out, "CNN Autoencoder"),
                use_container_width=True,
            )
        else:
            st.warning("CNN model not loaded.")

    st.markdown("---")
    st.subheader(f"Quantitative Metrics ({mode_option} Mode)")

    import pandas as pd

    def metrics_row(name, arr, latency_ms):
        psnr = calculate_psnr(clean_np, arr)
        ssim = calculate_ssim(clean_np, arr)
        psnr_text = f"{psnr:.2f}" if psnr != float("inf") else "inf"

        return {
            "Method": name,
            "PSNR (dB)": psnr_text,
            "SSIM": f"{ssim:.4f}",
            "Latency (ms)": f"{latency_ms:.1f}",
        }

    rows = [
        metrics_row("Gaussian Filter", gaussian_out, gaussian_ms),
        metrics_row("Median Filter", median_out, median_ms),
        metrics_row("Bilateral Filter", bilateral_out, bilateral_ms),
    ]

    if model is not None:
        rows.append(metrics_row("CNN Autoencoder", cnn_out, cnn_ms))

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader(
        "Absolute Error Maps | |Denoised - Clean| (amplified x5)"
    )
    error_columns = st.columns(4)

    def error_map_image(arr):
        error = np.abs(
            arr.astype(np.float64) - clean_np.astype(np.float64)
        )
        if error.ndim == 3:
            error = np.mean(error, axis=2)
        amplified = np.clip(error * 5.0, 0.0, 1.0)
        return Image.fromarray(
            (amplified * 255).astype(np.uint8), mode="L"
        )

    with error_columns[0]:
        st.image(
            error_map_image(gaussian_out),
            caption="Gaussian Error",
            use_container_width=True,
        )
    with error_columns[1]:
        st.image(
            error_map_image(median_out),
            caption="Median Error",
            use_container_width=True,
        )
    with error_columns[2]:
        st.image(
            error_map_image(bilateral_out),
            caption="Bilateral Error",
            use_container_width=True,
        )
    with error_columns[3]:
        if model is not None:
            st.image(
                error_map_image(cnn_out),
                caption="CNN Error",
                use_container_width=True,
            )

    st.markdown("---")
    st.caption(
        "B.Tech Digital Image Processing Project: Deep Learning-Based "
        "Image Denoising Using CNN Autoencoders. The CNN was trained on "
        "natural image patches (grayscale replicated across three channels). "
        "In Colour mode, luminance (Y) is denoised while chrominance (Cr/Cb) "
        "is preserved. Results use actual model inference."
    )


if __name__ == "__main__":
    main()