"""P1-01 集成测试：scripts/train.py --resume 精确续训。

验证训练产物契约：
- ``checkpoint_best.pt`` / ``checkpoint_last.pt``：完整训练状态
  （model/optimizer/scheduler/scaler/RNG），供 ``--resume`` 精确续训；
- ``best_model.pt``：裸 ``state_dict`` 推理 artifact（predict.py / API 可
  weights_only 安全加载）。
"""

import subprocess
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_smoke_config(tmp_path: Path) -> Path:
    config = {
        "data": {
            "max_sequence_length": 50,
            "max_ptm_sites": 5,
            "label_column": "cell_state",
            "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
            "num_workers": 0,
        },
        "model": {
            "encoder_type": "cnn",
            "embedding_dim": 16,
            "hidden_dim": 16,
            "num_layers": 1,
            "kernel_size": 3,
        },
        "training": {
            "max_epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_workers": 0,
            "seed": 42,
            "scheduler": "step",
            "scheduler_params": {"step_size": 1, "gamma": 0.5},
        },
        "release_gate": {"thresholds": {"accuracy": 0.0}},
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run_train(tmp_path, output, *, extra=None):
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        str(_write_smoke_config(tmp_path)),
        "--output",
        str(output),
        "--seed",
        "42",
    ]
    if extra:
        cmd.extend(extra)
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )


def test_train_artifacts_split_training_state_from_inference_artifact(tmp_path):
    """完整状态在 checkpoint_best.pt，best_model.pt 为裸 state_dict。"""
    output = tmp_path / "out"
    completed = _run_train(tmp_path, output)
    assert completed.returncode == 0, completed.stderr

    models = output / "models"
    assert (models / "checkpoint_best.pt").exists()
    assert (models / "checkpoint_best_last.pt").exists()

    ckpt = torch.load(
        models / "checkpoint_best.pt", map_location="cpu", weights_only=False
    )
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt, "optimizer 状态必须写入 checkpoint"
    assert "scheduler_state_dict" in ckpt, "scheduler 状态必须写入 checkpoint"
    assert "rng_state" in ckpt, "RNG 状态必须写入 checkpoint"
    assert ckpt["rng_state"]["torch_cpu"].numel() > 0

    # best_model.pt 必须是裸 state_dict（无 model_state_dict 包装键）
    infer = torch.load(models / "best_model.pt", map_location="cpu", weights_only=True)
    assert "model_state_dict" not in infer
    assert any(key.endswith("weight") or key.endswith("bias") for key in infer)
    for key, value in ckpt["model_state_dict"].items():
        assert torch.equal(infer[key], value), f"best_model.pt 未导出最佳权重: {key}"


def test_train_resume_continues_from_saved_epoch(tmp_path):
    """resume 后训练应从已完成的 epoch 继续（不从头重训）。"""
    output = tmp_path / "out"
    first = _run_train(tmp_path, output, extra=["--epochs", "1"])
    assert first.returncode == 0, first.stderr

    ckpt = torch.load(
        output / "models" / "checkpoint_best.pt", map_location="cpu", weights_only=False
    )
    first_epoch = ckpt["epoch"]

    second = _run_train(
        tmp_path,
        output,
        extra=["--epochs", "3", "--resume", str(output / "models" / "checkpoint_best.pt")],
    )
    assert second.returncode == 0, second.stderr
    assert "已恢复 optimizer 状态" in second.stdout, second.stdout
    assert "已恢复 scheduler 状态" in second.stdout, second.stdout
    assert "下一轮起始 epoch=1" in second.stdout, second.stdout

    # TD-M01: resume 后 best_value 恢复，val_loss 未改善时 checkpoint_best.pt
    # 不被覆盖；以 checkpoint_best_last.pt（每 epoch 保存）验证 epoch 推进。
    final_ckpt = torch.load(
        output / "models" / "checkpoint_best_last.pt", map_location="cpu", weights_only=False
    )
    # resume 后训练的 epoch 从 first_epoch 继续，总 epoch 数 = 3（0,1,2）
    assert final_ckpt["epoch"] >= first_epoch + 1


def test_train_resume_does_not_replay_completed_epoch(tmp_path):
    """resume 到 max_epochs=2 时只运行未完成的 epoch 1。"""
    output = tmp_path / "out"
    first = _run_train(tmp_path, output, extra=["--epochs", "1"])
    assert first.returncode == 0, first.stderr

    second = _run_train(
        tmp_path,
        output,
        extra=["--epochs", "2", "--resume", str(output / "models" / "checkpoint_best.pt")],
    )
    assert second.returncode == 0, second.stderr
    assert "下一轮起始 epoch=1" in second.stdout, second.stdout
    assert "Epoch   0:" not in second.stdout, second.stdout
    assert "Epoch   1:" in second.stdout, second.stdout


def test_train_resume_missing_checkpoint_fails_fast(tmp_path):
    """resume 不存在的 checkpoint 应显式失败。"""
    output = tmp_path / "out"
    completed = _run_train(
        tmp_path,
        output,
        extra=["--epochs", "1", "--resume", str(tmp_path / "does_not_exist.pt")],
    )
    assert completed.returncode != 0
    assert "checkpoint 不存在" in (completed.stderr + completed.stdout)


def test_train_resume_legacy_best_model_still_works(tmp_path):
    """旧版 best_model.pt（完整 dict）仍可作为 resume 输入（兼容路径）。"""
    output = tmp_path / "out"
    first = _run_train(tmp_path, output, extra=["--epochs", "1"])
    assert first.returncode == 0, first.stderr

    # 用旧命名模拟用户把 checkpoint_best 重命名为 best_model 的兼容场景：
    # 直接把完整 checkpoint 复制为 legacy.pt 并 resume 它
    legacy = tmp_path / "legacy.pt"
    import shutil

    shutil.copy(output / "models" / "checkpoint_best.pt", legacy)

    second = _run_train(
        tmp_path,
        tmp_path / "out2",
        extra=["--epochs", "2", "--resume", str(legacy)],
    )
    assert second.returncode == 0, second.stderr
    assert "已恢复 optimizer 状态" in second.stdout


def test_train_checkpoint_contains_callback_states_and_resume_restores(tmp_path):
    """TD-M01: checkpoint 序列化 callback 状态，resume 后恢复（早停语义连续）。"""
    output = tmp_path / "out"
    first = _run_train(tmp_path, output, extra=["--epochs", "1"])
    assert first.returncode == 0, first.stderr

    ckpt = torch.load(
        output / "models" / "checkpoint_best.pt", map_location="cpu", weights_only=False
    )
    assert "callback_states" in ckpt, "checkpoint 必须包含 callback_states"
    assert "ModelCheckpoint" in ckpt["callback_states"]
    assert "EarlyStopping" in ckpt["callback_states"]
    mc_state = ckpt["callback_states"]["ModelCheckpoint"]
    assert mc_state["monitor"] == "val_loss" and mc_state["mode"] == "min"
    es_state = ckpt["callback_states"]["EarlyStopping"]
    assert es_state["monitor"] == "val_loss" and es_state["mode"] == "min"

    second = _run_train(
        tmp_path,
        output,
        extra=["--epochs", "2", "--resume", str(output / "models" / "checkpoint_best.pt")],
    )
    assert second.returncode == 0, second.stderr
    assert "已恢复 callback 状态: ModelCheckpoint" in second.stdout, second.stdout
    assert "已恢复 callback 状态: EarlyStopping" in second.stdout, second.stdout
    assert "已恢复 optimizer 状态" in second.stdout, second.stdout


def test_train_resume_callback_state_mismatch_warns_but_continues(tmp_path):
    """TD-M01: callback 状态与配置不一致时告警并继续（兼容缺省路径）。"""
    output = tmp_path / "out"
    first = _run_train(tmp_path, output, extra=["--epochs", "1"])
    assert first.returncode == 0, first.stderr

    # 修改 config 使 patience 变化，触发 EarlyStopping 状态校验失败
    config_path = _write_smoke_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["early_stopping_patience"] = 999
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    second = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "out2"),
            "--seed",
            "42",
            "--epochs",
            "2",
            "--resume",
            str(output / "models" / "checkpoint_best.pt"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )
    assert second.returncode == 0, second.stderr
    assert "EarlyStopping 状态恢复失败" in second.stdout, second.stdout
