"""
数据准备脚本
功能: 生成示例数据，执行完整的预处理流程
"""

import os
import sys


def _ensure_project_root() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def main():
    """主函数"""
    _ensure_project_root()
    from src.utils.config import Config
    from src.utils.logging import setup_logger, get_timestamped_log_filename
    from src.utils.io import save_dataframe
    from src.data.loaders import DataLoader
    from src.data.preprocess import DataPreprocessor

    logger = setup_logger(__name__, get_timestamped_log_filename("prepare_data"))
    logger.info("=" * 60)
    logger.info("开始数据准备流程")
    logger.info("=" * 60)

    # DEMO 警告（审计 P1-2）：load_sample_data 生成的是随机序列 + 随机标签，
    # 仅用于冒烟测试。在此显式提醒用户，避免把合成数据当作真实数据训练。
    logger.warning(
        "注意：load_sample_data() 生成的是随机合成数据（随机序列 + 随机标签），"
        "仅用于端到端链路冒烟测试，不代表真实生物学分布。"
        "如需真实预测能力，请按 docs/guides/data_integration.md 准备真实数据。"
    )

    config = Config.from_yaml("configs/default.yaml")

    logger.info("步骤 1: 生成示例数据")
    loader = DataLoader(config.to_dict())
    df = loader.load_sample_data(num_samples=500)

    logger.info("步骤 2: 保存原始数据")
    save_dataframe(df, "data/raw/sample_data.csv")
    logger.info("原始数据已保存: data/raw/sample_data.csv")

    logger.info("步骤 3: 数据预处理")
    preprocessor = DataPreprocessor(config.to_dict())
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

    logger.info("步骤 4: 保存处理后的数据")
    save_dataframe(train_df, "data/processed/train.csv")
    save_dataframe(val_df, "data/processed/val.csv")
    save_dataframe(test_df, "data/processed/test.csv")
    logger.info("训练集: %d 条", len(train_df))
    logger.info("验证集: %d 条", len(val_df))
    logger.info("测试集: %d 条", len(test_df))

    logger.info("=" * 60)
    logger.info("数据准备流程完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
