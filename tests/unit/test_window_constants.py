"""Window-size constant guards (N05, 2026-08-16).

PTM-site windows (training, inference, reporting) previously hard-coded
31/15/16 magic numbers in four places, risking silent train/inference
window drift. All call sites now import DEFAULT_PTM_WINDOW_SIZE /
DEFAULT_PTM_HALF_WINDOW from src.data.aa_constants; these tests:

1. assert the constants are self-consistent (odd width, centred site);
2. assert model/dataset default signatures cannot drift from the constant;
3. assert the reporting-slice formula and the training-window formula
   extract identical windows (semantic equivalence regression);
4. assert the batch predictor script no longer carries the magic numbers.
"""

import inspect

import pytest

from src.data.aa_constants import (
    DEFAULT_PTM_WINDOW_SIZE,
    DEFAULT_PTM_HALF_WINDOW,
    AA_PAD_CHAR,
)
from src.data.ptm_site_dataset import PTMSiteDataset, PTMSiteDataModule
from src.models.ptm_site_predictor import PTMSitePredictor
from src.models.multitask_ptm import MultiTaskPTMPredictor

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_constants_are_self_consistent() -> None:
    """N05: 窗口必须为奇数且位点居中（half = (width-1)//2）。"""
    assert DEFAULT_PTM_WINDOW_SIZE == 2 * DEFAULT_PTM_HALF_WINDOW + 1
    assert DEFAULT_PTM_WINDOW_SIZE % 2 == 1


def test_model_defaults_match_constant() -> None:
    """N05: 模型与数据集默认 window_size 必须等于常量（防漂移）。"""
    for cls in (PTMSiteDataset, PTMSiteDataModule, PTMSitePredictor, MultiTaskPTMPredictor):
        sig = inspect.signature(cls.__init__)
        default = sig.parameters["window_size"].default
        assert default == DEFAULT_PTM_WINDOW_SIZE, (
            f"{cls.__name__}.__init__ window_size default {default} drifted "
            f"from DEFAULT_PTM_WINDOW_SIZE={DEFAULT_PTM_WINDOW_SIZE}"
        )


def _training_window(sequence: str, pos0: int) -> str:
    """训练侧窗口公式（0-indexed 位点，与 _encode_sequence_window 同构）。"""
    hw = DEFAULT_PTM_HALF_WINDOW
    start = max(0, pos0 - hw)
    end = min(len(sequence), pos0 + hw + 1)
    window = sequence[start:end]
    if len(window) < DEFAULT_PTM_WINDOW_SIZE:
        pad_left = max(0, hw - pos0)
        pad_right = max(0, (pos0 + hw + 1) - len(sequence))
        window = AA_PAD_CHAR * pad_left + window + AA_PAD_CHAR * pad_right
    return window


@pytest.mark.parametrize("seq_len", [31, 50, 100, 200])
@pytest.mark.parametrize("pos0", [0, 1, 7, 15, 30, 99])
def test_report_slice_matches_training_window(seq_len: int, pos0: int) -> None:
    """N05: 报告窗口切片（1-indexed pos = pos0+1）与训练窗口提取等价。

    仅对序列长度充足（无需 padding）的用例断言切片相等；padding 语义
    属训练输入固定宽度要求，展示字段保留真实切片属预期差异。
    """
    if pos0 >= seq_len:
        pytest.skip("position outside sequence")
    sequence = "ACDEFGHIKLMNPQRSTVWY" * (seq_len // 20 + 1)
    sequence = sequence[:seq_len]

    train = _training_window(sequence, pos0)
    report = sequence[
        max(0, (pos0 + 1) - (DEFAULT_PTM_HALF_WINDOW + 1)):
        (pos0 + 1) + DEFAULT_PTM_HALF_WINDOW
    ]
    if len(sequence) - DEFAULT_PTM_HALF_WINDOW > pos0 >= DEFAULT_PTM_HALF_WINDOW:
        assert report == train
        # 位点必须位于窗口正中
        assert report[DEFAULT_PTM_HALF_WINDOW] == sequence[pos0]
    # 边界外（序列两端）不崩溃且不越界
    assert len(report) <= DEFAULT_PTM_WINDOW_SIZE


def test_predict_script_no_magic_window_numbers() -> None:
    """N05: predict_ptm_sites.py 中不得残留窗口魔法数字面量。"""
    script = (PROJECT_ROOT / "scripts" / "predict_ptm_sites.py").read_text(encoding="utf-8")
    for magic in ("self.window_size = 31", "self.half_window = 15", "position'-16", "position+15"):
        assert magic not in script, f"magic window literal still present: {magic!r}"
