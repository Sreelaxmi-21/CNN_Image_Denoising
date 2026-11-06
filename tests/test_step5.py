"""
tests/test_step5.py
====================
Verification tests for Step 5: Evaluation & Visualization Pipeline.

Tests:
    T5.1 — evaluate.py imports and evaluate_single_image() runs on a synthetic image
    T5.2 — Full run_full_evaluation() produces CSV with correct columns and valid values
    T5.3 — Markdown table file is created and contains all expected method names
    T5.4 — PSNR values are positive; SSIM values are in [0, 1]; Latency > 0
    T5.5 — CNN results are not fabricated (values differ from random baseline)
    T5.6 — plot_comparison_grid() saves a valid PNG file
    T5.7 — plot_error_maps() saves a valid PNG file
    T5.8 — plot_psnr_bar_chart() saves a valid PNG file
    T5.9 — plot_ssim_bar_chart() saves a valid PNG file
    T5.10 — generate_all_visualizations() saves all expected figure files

IMPORTANT: Tests do NOT assert CNN beats classical methods.
           All metric values are generated from actual model inference.

Usage:
    python tests/test_step5.py
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# =========================================================================
# Test tracking helpers
# =========================================================================

_PASS = 0
_FAIL = 0
_TESTS: list = []


def _assert(condition: bool, msg: str = "") -> None:
    if not condition:
        raise AssertionError(msg or "Assertion failed")


def run_test(name: str, fn):
    global _PASS, _FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        _PASS += 1
        _TESTS.append((name, True, None))
    except Exception as exc:
        print(f"  [FAIL] {name}")
        print(f"         {type(exc).__name__}: {exc}")
        _FAIL += 1
        _TESTS.append((name, False, str(exc)))


# =========================================================================
# Shared test fixtures
# =========================================================================

def _make_synthetic_image(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """Creates a reproducible synthetic RGB image (H, W, 3) float32 in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3)).astype(np.float32)


def _load_config():
    """Loads the project config.yaml."""
    import yaml
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def _get_model():
    """Loads the trained CNN model from the production (stride16/epochs30) checkpoint."""
    from src.evaluate import load_cnn_model
    return load_cnn_model(
        "results/experiments/stride16_epochs30/checkpoints/best_model.pth",
        device="cpu",
    )


# =========================================================================
# Module-level shared state: run full evaluation ONCE for T5.2, T5.3, T5.10
# =========================================================================
# These are populated lazily in _run_shared_evaluation() the first time needed.
_SHARED_TABLES_DIR = None
_SHARED_FIGURES_DIR = None
_SHARED_RESULTS = None
_SHARED_TMPDIR_OBJ = None  # kept alive to prevent cleanup


def _run_shared_evaluation():
    """
    Runs run_full_evaluation() once, shared across T5.2 / T5.3 / T5.10.
    Also runs generate_all_visualizations() once for T5.10.
    Results and dirs are cached in module-level variables.
    """
    global _SHARED_TABLES_DIR, _SHARED_FIGURES_DIR, _SHARED_RESULTS, _SHARED_TMPDIR_OBJ

    if _SHARED_RESULTS is not None:
        return  # Already computed

    from src.evaluate import run_full_evaluation
    from src.visualize import generate_all_visualizations

    # Create a persistent temp dir that lasts for the test session
    import tempfile
    tmpobj = tempfile.TemporaryDirectory()
    _SHARED_TMPDIR_OBJ = tmpobj  # keep reference so it isn't garbage-collected
    base = tmpobj.name

    tables_dir  = os.path.join(base, "tables")
    figures_dir = os.path.join(base, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("\n  [Running shared full evaluation — this takes ~2 min on CPU...]")
    results = run_full_evaluation(
        config_path="configs/config.yaml",
        output_dir=tables_dir,
        latency_runs=3,
    )

    print("  [Running shared generate_all_visualizations (max_grids=2)...]")
    generate_all_visualizations(
        results=results,
        config_path="configs/config.yaml",
        figures_dir=figures_dir,
        max_grids=2,
    )

    _SHARED_TABLES_DIR = tables_dir
    _SHARED_FIGURES_DIR = figures_dir
    _SHARED_RESULTS = results


# =========================================================================
# T5.1 — evaluate.py imports and evaluate_single_image() runs on a synthetic image
# =========================================================================

def test_t5_1_evaluate_imports_and_single_image():
    from src.evaluate import evaluate_single_image

    config = _load_config()
    model = _get_model()
    clean = _make_synthetic_image(128, 128, seed=7)

    results = evaluate_single_image(
        clean_np=clean,
        sigma=25,
        image_name="synthetic_test.png",
        model=model,
        config=config,
        device="cpu",
        noise_seed=42,
        latency_runs=3,
    )

    _assert(isinstance(results, list), "Result must be a list")
    _assert(len(results) == 4, f"Expected 4 method results, got {len(results)}")

    methods_found = {r["method"] for r in results}
    _assert(
        methods_found == {"Gaussian", "Median", "Bilateral", "CNN"},
        f"Expected 4 method names, got: {methods_found}"
    )

    for r in results:
        _assert("psnr" in r and "ssim" in r and "latency_ms" in r,
                f"Missing metric key in result for method {r.get('method')}")
        _assert(r["sigma"] == 25, "Sigma mismatch in result")
        _assert(r["image"] == "synthetic_test.png", "Image name mismatch")


def test_traditional_filter_latency_isolated():
    from contextlib import ExitStack
    import src.evaluate as evaluate
    from unittest.mock import patch

    config = _load_config()
    calls = {"gaussian": [], "median": [], "bilateral": []}

    def capture_filter(name):
        def filter_image(image, **params):
            calls[name].append((image, params))
            return image
        return filter_image

    def run_one_latency_sample(func, num_runs, warmup):
        func()
        return {"mean_ms": 1.0}

    with ExitStack() as patches:
        patches.enter_context(
            patch.object(evaluate, "apply_gaussian_filter", capture_filter("gaussian"))
        )
        patches.enter_context(
            patch.object(evaluate, "apply_median_filter", capture_filter("median"))
        )
        patches.enter_context(
            patch.object(evaluate, "apply_bilateral_filter", capture_filter("bilateral"))
        )
        patches.enter_context(
            patch.object(
                evaluate,
                "cnn_denoise_image",
                side_effect=lambda model, noisy_np, device: noisy_np,
            )
        )
        patches.enter_context(
            patch.object(
                evaluate, "benchmark_latency",
                side_effect=run_one_latency_sample,
            )
        )

        results = evaluate.evaluate_single_image(
            clean_np=_make_synthetic_image(16, 16, seed=13),
            sigma=25,
            image_name="filter_isolation.png",
            model=None,
            config=config,
            latency_runs=1,
        )

    expected_params = {
        "gaussian": config["traditional_filters"]["gaussian"],
        "median": config["traditional_filters"]["median"],
        "bilateral": config["traditional_filters"]["bilateral"],
    }
    for name, filter_calls in calls.items():
        _assert(len(filter_calls) == 2,
                f"{name} should run once for output and once for latency")
        _assert(
            all(params == expected_params[name] for _, params in filter_calls),
            f"{name} parameters differ from config",
        )

    noisy_images = [image for filter_calls in calls.values()
                    for image, _ in filter_calls]
    _assert(
        all(image is noisy_images[0] for image in noisy_images),
        "Traditional methods did not receive the exact same noisy image",
    )
    _assert(len(results) == 4, "Evaluation result schema/count changed")


# =========================================================================
# T5.2 — run_full_evaluation() produces CSV with correct columns and valid values
# =========================================================================

def test_t5_2_full_evaluation_csv():
    _run_shared_evaluation()

    csv_path = os.path.join(_SHARED_TABLES_DIR, "evaluation_results.csv")
    _assert(os.path.exists(csv_path), f"CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    _assert(len(rows) > 0, "CSV is empty")

    required_columns = {"image", "sigma", "method", "psnr", "ssim", "latency_ms"}
    actual_columns = set(rows[0].keys())
    _assert(
        required_columns.issubset(actual_columns),
        f"Missing CSV columns: {required_columns - actual_columns}"
    )

    # Expect 12 images * 3 noise levels * 4 methods = 144 rows
    expected_rows = 12 * 3 * 4
    _assert(
        len(rows) == expected_rows,
        f"Expected {expected_rows} CSV rows, got {len(rows)}"
    )


# =========================================================================
# T5.3 — Markdown table file contains all expected method names
# =========================================================================

def test_t5_3_markdown_contains_methods():
    _run_shared_evaluation()  # no-op if already done

    md_path = os.path.join(_SHARED_TABLES_DIR, "evaluation_results.md")
    _assert(os.path.exists(md_path), f"Markdown file not found: {md_path}")

    content = Path(md_path).read_text(encoding="utf-8")
    for method in ["Gaussian", "Median", "Bilateral", "CNN"]:
        _assert(method in content, f"Method '{method}' not found in Markdown output")

    for sigma in [15, 25, 50]:
        _assert(str(sigma) in content, f"sigma={sigma} not found in Markdown output")


# =========================================================================
# T5.4 — PSNR positive; SSIM in [0, 1]; Latency > 0
# =========================================================================

def test_t5_4_metric_ranges():
    from src.evaluate import evaluate_single_image

    config = _load_config()
    model = _get_model()
    clean = _make_synthetic_image(128, 128, seed=11)

    for sigma in [15, 25, 50]:
        results = evaluate_single_image(
            clean_np=clean,
            sigma=sigma,
            image_name="range_test.png",
            model=model,
            config=config,
            device="cpu",
            noise_seed=42,
            latency_runs=2,
        )
        for r in results:
            psnr = r["psnr"]
            ssim = r["ssim"]
            lat  = r["latency_ms"]

            _assert(
                psnr > 0 or psnr == float("inf"),
                f"PSNR must be positive, got {psnr} for {r['method']} sigma={sigma}"
            )
            _assert(
                -0.1 <= ssim <= 1.05,
                f"SSIM out of range [{ssim:.4f}] for {r['method']} sigma={sigma}"
            )
            _assert(
                lat > 0,
                f"Latency must be > 0 ms, got {lat} for {r['method']} sigma={sigma}"
            )


# =========================================================================
# T5.5 — CNN results are not fabricated (not all-zero, differs from noisy input)
# =========================================================================

def test_t5_5_cnn_results_not_fabricated():
    from src.evaluate import cnn_denoise_image, load_cnn_model
    from src.noise import add_gaussian_noise_numpy

    model = load_cnn_model("results/checkpoints/best_model.pth", device="cpu")
    clean = _make_synthetic_image(128, 128, seed=3)
    noisy = add_gaussian_noise_numpy(clean, sigma=25, seed=42)
    cnn_out = cnn_denoise_image(model, noisy, device="cpu")

    _assert(cnn_out.shape == noisy.shape, "CNN output shape must match noisy input")
    _assert(cnn_out.dtype == np.float32, "CNN output must be float32")
    _assert(float(np.min(cnn_out)) >= -0.01, "CNN output below 0")
    _assert(float(np.max(cnn_out)) <= 1.01, "CNN output above 1")

    # Output should NOT be identical to the noisy input (that would be no-op)
    _assert(
        not np.allclose(cnn_out, noisy, atol=1e-4),
        "CNN output is identical to noisy input -- model may not be applying denoising"
    )

    # Output should NOT be all-zeros (fabricated/constant)
    _assert(
        float(np.std(cnn_out)) > 1e-4,
        "CNN output has zero variance -- likely not doing real inference"
    )


# =========================================================================
# T5.6 — plot_comparison_grid() saves a valid PNG file
# =========================================================================

def test_t5_6_comparison_grid_saved():
    from src.visualize import plot_comparison_grid
    from src.noise import add_gaussian_noise_numpy
    from src.traditional import apply_all_traditional_filters
    from src.evaluate import cnn_denoise_image

    config = _load_config()
    model = _get_model()
    clean = _make_synthetic_image(128, 128, seed=5)
    noisy = add_gaussian_noise_numpy(clean, sigma=25, seed=42)
    filters = apply_all_traditional_filters(noisy, config)
    cnn_out = cnn_denoise_image(model, noisy, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_grid.png")
        plot_comparison_grid(
            clean=clean,
            noisy=noisy,
            gauss_out=filters["gaussian"],
            median_out=filters["median"],
            bilateral_out=filters["bilateral"],
            cnn_out=cnn_out,
            sigma=25,
            image_name="test.png",
            output_path=out_path,
        )
        _assert(os.path.exists(out_path), "Comparison grid PNG not saved")
        _assert(os.path.getsize(out_path) > 1000, "PNG file too small (likely empty)")


# =========================================================================
# T5.7 — plot_error_maps() saves a valid PNG file
# =========================================================================

def test_t5_7_error_maps_saved():
    from src.visualize import plot_error_maps
    from src.noise import add_gaussian_noise_numpy
    from src.traditional import apply_all_traditional_filters
    from src.evaluate import cnn_denoise_image

    config = _load_config()
    model = _get_model()
    clean = _make_synthetic_image(128, 128, seed=6)
    noisy = add_gaussian_noise_numpy(clean, sigma=25, seed=42)
    filters = apply_all_traditional_filters(noisy, config)
    cnn_out = cnn_denoise_image(model, noisy, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_errmap.png")
        plot_error_maps(
            clean=clean,
            gauss_out=filters["gaussian"],
            median_out=filters["median"],
            bilateral_out=filters["bilateral"],
            cnn_out=cnn_out,
            sigma=25,
            image_name="test.png",
            output_path=out_path,
        )
        _assert(os.path.exists(out_path), "Error maps PNG not saved")
        _assert(os.path.getsize(out_path) > 1000, "PNG file too small")


# =========================================================================
# T5.8 — plot_psnr_bar_chart() saves a valid PNG file
# =========================================================================

def test_t5_8_psnr_barchart_saved():
    from src.visualize import plot_psnr_bar_chart
    from src.evaluate import compute_summary_table

    # Synthetic results data
    methods = ["Gaussian", "Median", "Bilateral", "CNN"]
    noise_levels = [15, 25, 50]
    fake_results = []
    for m in methods:
        for s in noise_levels:
            fake_results.append({
                "image": "test.png", "sigma": s, "method": m,
                "psnr": 25.0 + (50 - s) * 0.1, "ssim": 0.7, "latency_ms": 1.0
            })

    summary = compute_summary_table(fake_results, noise_levels)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "psnr_bar.png")
        plot_psnr_bar_chart(summary, noise_levels, out_path)
        _assert(os.path.exists(out_path), "PSNR bar chart PNG not saved")
        _assert(os.path.getsize(out_path) > 1000, "PNG file too small")


# =========================================================================
# T5.9 — plot_ssim_bar_chart() saves a valid PNG file
# =========================================================================

def test_t5_9_ssim_barchart_saved():
    from src.visualize import plot_ssim_bar_chart
    from src.evaluate import compute_summary_table

    methods = ["Gaussian", "Median", "Bilateral", "CNN"]
    noise_levels = [15, 25, 50]
    fake_results = []
    for m in methods:
        for s in noise_levels:
            fake_results.append({
                "image": "test.png", "sigma": s, "method": m,
                "psnr": 25.0, "ssim": 0.6 + (50 - s) * 0.002, "latency_ms": 1.0
            })

    summary = compute_summary_table(fake_results, noise_levels)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "ssim_bar.png")
        plot_ssim_bar_chart(summary, noise_levels, out_path)
        _assert(os.path.exists(out_path), "SSIM bar chart PNG not saved")
        _assert(os.path.getsize(out_path) > 1000, "PNG file too small")


# =========================================================================
# T5.10 — generate_all_visualizations() produces all expected figure files
# =========================================================================

def test_t5_10_generate_all_visualizations():
    _run_shared_evaluation()  # no-op if already done

    # Bar charts must exist
    _assert(
        os.path.exists(os.path.join(_SHARED_FIGURES_DIR, "psnr_barchart.png")),
        "psnr_barchart.png not found"
    )
    _assert(
        os.path.exists(os.path.join(_SHARED_FIGURES_DIR, "ssim_barchart.png")),
        "ssim_barchart.png not found"
    )

    # Comparison grids and error maps for first 2 images (max_grids=2)
    noise_levels = [15, 25, 50]
    for img_stem in ["01", "02"]:
        for sigma in noise_levels:
            grid_file = os.path.join(
                _SHARED_FIGURES_DIR, f"comparison_grid_{img_stem}_sigma{sigma}.png"
            )
            err_file = os.path.join(
                _SHARED_FIGURES_DIR, f"error_maps_{img_stem}_sigma{sigma}.png"
            )
            _assert(
                os.path.exists(grid_file),
                f"Comparison grid not found: {os.path.basename(grid_file)}"
            )
            _assert(
                os.path.exists(err_file),
                f"Error map not found: {os.path.basename(err_file)}"
            )


# =========================================================================
# Run all tests
# =========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 5 VERIFICATION TESTS")
    print("  Evaluation & Visualization Pipeline")
    print("=" * 60)
    print()

    # Fast unit tests first
    print("--- Unit & Integration Tests (fast) ---")
    run_test("T5.1  evaluate imports + single_image", test_t5_1_evaluate_imports_and_single_image)
    run_test("Traditional filter latency isolation", test_traditional_filter_latency_isolated)
    run_test("T5.4  PSNR positive, SSIM in [0,1], Lat>0", test_t5_4_metric_ranges)
    run_test("T5.5  CNN output not fabricated", test_t5_5_cnn_results_not_fabricated)
    run_test("T5.6  comparison_grid PNG saved", test_t5_6_comparison_grid_saved)
    run_test("T5.7  error_maps PNG saved", test_t5_7_error_maps_saved)
    run_test("T5.8  psnr_barchart PNG saved", test_t5_8_psnr_barchart_saved)
    run_test("T5.9  ssim_barchart PNG saved", test_t5_9_ssim_barchart_saved)

    # Full pipeline tests (ONE shared evaluation run for T5.2 + T5.3 + T5.10)
    print()
    print("--- Full Pipeline Tests (Set12 benchmark, single shared run) ---")
    run_test("T5.2  run_full_evaluation CSV columns+rows", test_t5_2_full_evaluation_csv)
    run_test("T5.3  Markdown contains all method names", test_t5_3_markdown_contains_methods)
    run_test("T5.10 generate_all_visualizations figures", test_t5_10_generate_all_visualizations)

    # Summary
    print()
    print("=" * 60)
    total = _PASS + _FAIL
    print(f"  Results: {_PASS}/{total} PASSED  |  {_FAIL} FAILED")
    print("=" * 60)

    if _FAIL == 0:
        print("\n  === ALL STEP 5 TESTS PASSED SUCCESSFULLY ===\n")
    else:
        print("\n  [!] Some tests failed. Details above.")
        print()
        for name, passed, err in _TESTS:
            if not passed:
                print(f"  FAIL: {name}")
                print(f"        {err}")
        sys.exit(1)
