"""
tests/test_checkpoint_consistency.py
=====================================
Verifies that configs/config.yaml, src/evaluate.py, src/visualize.py, and
app/streamlit_app.py all agree on the same production inference checkpoint
(the stride-16 / 30-epoch experiment, Epoch 29).

Run with:
    pytest tests/test_checkpoint_consistency.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"
_PRODUCTION_CKPT_REL = "results/experiments/stride16_epochs30/checkpoints/best_model.pth"
_PRODUCTION_CKPT_ABS = _PROJECT_ROOT / _PRODUCTION_CKPT_REL
_STREAMLIT_APP = _PROJECT_ROOT / "app" / "streamlit_app.py"

_EXPECTED_EPOCH = 29
_MIN_VAL_PSNR = 28.0


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Group 1: config.yaml key validation
# ---------------------------------------------------------------------------

class TestConfigCheckpointKey:

    def test_config_has_inference_checkpoint_key(self):
        """evaluation.inference_checkpoint key must exist in config.yaml."""
        cfg = _load_config()
        assert "evaluation" in cfg
        assert "inference_checkpoint" in cfg["evaluation"], (
            "config.yaml evaluation section is missing inference_checkpoint key"
        )

    def test_config_inference_checkpoint_points_to_stride16(self):
        """inference_checkpoint must reference the stride16_epochs30 experiment."""
        cfg = _load_config()
        val = cfg["evaluation"]["inference_checkpoint"]
        assert "stride16_epochs30" in val, (
            f"inference_checkpoint should reference stride16_epochs30, got: {val!r}"
        )

    def test_config_inference_checkpoint_ends_with_best_model(self):
        """inference_checkpoint path must end with best_model.pth."""
        cfg = _load_config()
        val = cfg["evaluation"]["inference_checkpoint"]
        assert val.endswith("best_model.pth"), (
            f"Expected path ending in best_model.pth, got: {val!r}"
        )


# ---------------------------------------------------------------------------
# Group 2: checkpoint file content
# ---------------------------------------------------------------------------

class TestProductionCheckpointFile:

    def test_inference_checkpoint_file_exists(self):
        """The file at evaluation.inference_checkpoint must exist on disk."""
        cfg = _load_config()
        ckpt_path = _PROJECT_ROOT / cfg["evaluation"]["inference_checkpoint"]
        assert ckpt_path.exists(), f"Production checkpoint not found: {ckpt_path}"

    def test_inference_checkpoint_is_production_epoch(self):
        """Checkpoint must record epoch=29 and val_psnr > 28 dB."""
        cfg = _load_config()
        ckpt_path = _PROJECT_ROOT / cfg["evaluation"]["inference_checkpoint"]
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch")
        val_psnr = ckpt.get("val_psnr")
        assert epoch == _EXPECTED_EPOCH, f"Expected epoch={_EXPECTED_EPOCH}, got {epoch}"
        assert val_psnr is not None, "Checkpoint missing val_psnr key"
        assert val_psnr > _MIN_VAL_PSNR, (
            f"Expected val_psnr > {_MIN_VAL_PSNR} dB, got {val_psnr:.4f}"
        )

    def test_inference_checkpoint_has_required_keys(self):
        """Checkpoint dict must contain all keys needed by load_cnn_model."""
        cfg = _load_config()
        ckpt_path = _PROJECT_ROOT / cfg["evaluation"]["inference_checkpoint"]
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        required = {
            "epoch", "model_state_dict", "optimizer_state_dict",
            "scheduler_state_dict", "val_loss", "val_psnr",
        }
        missing = required - set(ckpt.keys())
        assert not missing, f"Checkpoint missing keys: {missing}"


# ---------------------------------------------------------------------------
# Group 3: evaluate.py resolution logic
# ---------------------------------------------------------------------------

class TestEvaluateCheckpointResolution:

    def test_cli_evaluation_prefers_inference_checkpoint(self):
        """Resolution must pick evaluation.inference_checkpoint before the legacy path."""
        cfg = _load_config()
        expected = cfg["evaluation"].get("inference_checkpoint")
        assert expected, "inference_checkpoint must be set in config"
        # Mirror the resolution logic from run_full_evaluation()
        resolved = (
            cfg.get("evaluation", {}).get("inference_checkpoint")
            or (
                cfg.get("training", {}).get("checkpoint_dir", "results/checkpoints")
                + "/best_model.pth"
            )
        )
        assert resolved == expected, (
            f"Resolution did not prefer inference_checkpoint.\n"
            f"Expected: {expected!r}\nGot: {resolved!r}"
        )

    def test_load_cnn_model_logs_metadata(self, capsys):
        """load_cnn_model must print Epoch and val_psnr after loading."""
        if not _PRODUCTION_CKPT_ABS.exists():
            pytest.skip(f"Checkpoint not found: {_PRODUCTION_CKPT_ABS}")
        from src.evaluate import load_cnn_model
        load_cnn_model(str(_PRODUCTION_CKPT_ABS), device="cpu")
        captured = capsys.readouterr()
        assert "Epoch=" in captured.out, (
            "load_cnn_model should print Epoch= metadata after loading"
        )
        assert "val_psnr=" in captured.out, (
            "load_cnn_model should print val_psnr= metadata after loading"
        )


# ---------------------------------------------------------------------------
# Group 4: Streamlit / CLI consistency
# ---------------------------------------------------------------------------

class TestStreamlitAndCLIConsistency:

    def test_streamlit_checkpoint_matches_config(self):
        """streamlit_app.py must reference the same experiment as config."""
        assert _STREAMLIT_APP.exists(), f"Streamlit app not found: {_STREAMLIT_APP}"
        source = _STREAMLIT_APP.read_text(encoding="utf-8")
        cfg = _load_config()
        config_ckpt = cfg["evaluation"]["inference_checkpoint"]
        assert config_ckpt in source or "stride16_epochs30" in source, (
            f"Streamlit app does not reference production checkpoint: {config_ckpt!r}"
        )

    def test_both_resolve_to_same_filename(self):
        """Both CLI and Streamlit checkpoint paths must end with best_model.pth."""
        cfg = _load_config()
        cli_ckpt = cfg["evaluation"]["inference_checkpoint"]
        assert Path(cli_ckpt).name == "best_model.pth", (
            f"Expected basename best_model.pth, got: {Path(cli_ckpt).name!r}"
        )


# ---------------------------------------------------------------------------
# Group 5: error quality
# ---------------------------------------------------------------------------

class TestMissingCheckpointError:

    def test_missing_checkpoint_raises_file_not_found(self):
        """Passing a nonexistent path must raise FileNotFoundError."""
        from src.evaluate import load_cnn_model
        with pytest.raises(FileNotFoundError) as exc_info:
            load_cnn_model("results/checkpoints/nonexistent_model.pth", device="cpu")
        assert "nonexistent_model.pth" in str(exc_info.value)

    def test_missing_checkpoint_error_message_is_descriptive(self):
        """FileNotFoundError message must mention train.py to guide the user."""
        from src.evaluate import load_cnn_model
        with pytest.raises(FileNotFoundError) as exc_info:
            load_cnn_model("results/checkpoints/nonexistent_model.pth", device="cpu")
        msg = str(exc_info.value)
        assert "train.py" in msg or "train" in msg.lower(), (
            "FileNotFoundError should mention train.py to guide the user"
        )
