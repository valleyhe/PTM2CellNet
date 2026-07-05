"""端到端测试: PTM 位点预测训练 CLI。

对应缺口分析 §3.1 — Gap 3 (中等严重度):
PTM 位点预测训练 E2E 测试缺失。

运行最小 smoke 训练并验证导出产物存在。
"""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"


def _run(cmd, cwd=PROJECT_ROOT):
    """运行子进程，失败时打印完整输出。"""
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, (
        f"命令失败 (exit {result.returncode}): {' '.join(cmd)}"
    )
    return result


class TestPTMSiteTrainPredictE2E:
    """E2E 测试: PTM 位点训练 CLI 产物验证。"""

    def test_ptm_site_training_completes(self, tmp_path):
        """训练完成并产出 checkpoint。"""
        # 创建最小训练数据
        data_csv = tmp_path / "train.csv"
        rows = ["uniprot_id,sequence_window,position,aa,label,ptm_type"]
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY",
            "AYDEFGHIKLMNPQRSTVWC",
            "ACDEFGHIKLMNPQRSTVYA",
            "AKDEFGHIKLMNPQRSTVWC",
        ]
        for i, seq in enumerate(sequences):
            label = 1 if i % 2 == 0 else 0
            if label:
                position, aa, ptm_type = 3, "S", "phosphorylation"
            else:
                position, aa, ptm_type = 0, "X", ""
            rows.append(f'P{i:04d},{seq},{position},{aa},{label},{ptm_type}')

        data_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _run([
            sys.executable, str(SCRIPTS / "train_ptm_site.py"),
            "--data", str(data_csv),
            "--encoder", "cnn",
            "--embed-dim", "16",
            "--hidden-dim", "32",
            "--num-layers", "2",
            "--batch-size", "4",
            "--max-epochs", "1",
            "--output-dir", str(output_dir),
            "--gpus", "0",
        ])

        # 验证 checkpoint 已生成
        ckpts = list(output_dir.rglob("*.ckpt"))
        assert len(ckpts) >= 1, f"在 {output_dir} 中未找到 checkpoint 文件"


class TestPTMSiteEdgeCases:
    """PTM 位点训练边缘场景。"""

    def test_ptm_site_empty_ptm_column(self, tmp_path):
        """空 ptm_sites 列不应导致崩溃。"""
        data_csv = tmp_path / "train.csv"
        data_csv.write_text(
            "uniprot_id,sequence_window,position,aa,label,ptm_type\n"
            'P0001,ACDEFGHIKLMNPQRSTVWY,0,X,0,\n'
            'P0002,AYDEFGHIKLMNPQRSTVWC,0,X,1,phosphorylation\n',
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _run([
            sys.executable, str(SCRIPTS / "train_ptm_site.py"),
            "--data", str(data_csv),
            "--encoder", "cnn",
            "--embed-dim", "16",
            "--hidden-dim", "32",
            "--num-layers", "1",
            "--batch-size", "2",
            "--max-epochs", "1",
            "--output-dir", str(output_dir),
            "--gpus", "0",
        ])

        ckpts = list(output_dir.rglob("*.ckpt"))
        assert len(ckpts) >= 1
