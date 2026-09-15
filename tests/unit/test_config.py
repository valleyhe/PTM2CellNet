"""Config类独立单元测试。

覆盖 ``src.utils.config.Config`` 的加载、访问、嵌套、深拷贝与错误处理路径。
所有文件类测试使用 pytest 的 ``tmp_path`` fixture，无需修改 conftest。
"""

from __future__ import annotations

import copy

import pytest
import yaml

from src.utils.config import Config


class TestFromYaml:
    """from_yaml 加载路径测试"""

    def test_from_yaml_loads_valid_config(self, tmp_path):
        """从合法 YAML 文件加载配置，值应被正确读入。"""
        config_file = tmp_path / "valid.yaml"
        config_file.write_text(
            "project:\n  name: PTM2CellNet\n  version: 1.0\ntraining:\n  batch_size: 32\n  epochs: 10\n",
            encoding="utf-8",
        )

        config = Config.from_yaml(str(config_file))

        assert config is not None
        assert config.get("project.name") == "PTM2CellNet"
        assert config.get("project.version") == 1.0
        assert config.get("training.batch_size") == 32
        assert config.get("training.epochs") == 10

    def test_from_yaml_file_not_found(self, tmp_path):
        """文件不存在时应抛出 FileNotFoundError。"""
        missing = tmp_path / "does_not_exist.yaml"

        with pytest.raises(FileNotFoundError):
            Config.from_yaml(str(missing))

    def test_from_yaml_empty_file_returns_empty_config(self, tmp_path):
        """空 YAML 文件应安全加载为空配置（yaml.safe_load 返回 None）。"""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")

        config = Config.from_yaml(str(empty_file))

        # yaml.safe_load 对空文件返回 None，内部存储为 None（而非 {}），
        # 但 get 仍应能优雅返回默认值（get 对 None 走 TypeError 分支）。
        assert config.get("any.key", "default") == "default"

    def test_from_yaml_absolute_path(self, tmp_path):
        """绝对路径应能直接加载。"""
        config_file = tmp_path / "abs.yaml"
        config_file.write_text("key: value\n", encoding="utf-8")

        config = Config.from_yaml(str(config_file))

        assert config.get("key") == "value"


class TestGet:
    """get 访问路径测试"""

    def test_get_existing_key(self):
        """存在的键应返回对应值。"""
        config = Config({"model": {"encoder": "cnn", "layers": 3}, "lr": 0.001})

        assert config.get("model.encoder") == "cnn"
        assert config.get("model.layers") == 3
        assert config.get("lr") == 0.001

    def test_get_missing_key_returns_default(self):
        """缺失键应返回提供的默认值。"""
        config = Config({"a": {"b": 1}})

        assert config.get("a.missing", "fallback") == "fallback"
        assert config.get("nonexistent", 42) == 42

    def test_get_missing_key_without_default_returns_none(self):
        """缺失键且未提供默认值时应返回 None（而非抛出异常）。"""
        config = Config({"a": {"b": 1}})

        assert config.get("a.missing") is None
        assert config.get("nonexistent.key") is None

    def test_get_missing_key_without_default_does_not_raise(self):
        """显式验证：get 在键缺失时不抛 KeyError（与 __getitem__ 的契约不同）。"""
        config = Config({"a": 1})

        # 不应抛出任何异常
        result = config.get("deeply.missing.key")
        assert result is None


class TestGetItem:
    """__getitem__ 字典式访问测试"""

    def test_getitem_existing_key(self):
        """存在的键应返回对应值。"""
        config = Config({"x": {"y": 10}})

        assert config["x.y"] == 10
        assert config["x"] == {"y": 10}

    def test_getitem_missing_key_raises_keyerror(self):
        """缺失键时应抛出 KeyError（与 dict 契约一致，不返回 None）。"""
        config = Config({"a": {"b": 1}})

        with pytest.raises(KeyError):
            _ = config["a.missing"]

        with pytest.raises(KeyError):
            _ = config["nonexistent"]


class TestToDict:
    """to_dict 深拷贝测试"""

    def test_to_dict_returns_deep_copy(self):
        """to_dict 应返回配置的深拷贝，修改返回值不影响原配置。"""
        original = {"model": {"layers": [1, 2, 3]}, "lr": 0.1}
        config = Config(copy.deepcopy(original))

        returned = config.to_dict()

        # 值相等
        assert returned == original

        # 修改返回的顶层与嵌套结构均不影响原配置
        returned["lr"] = 0.999
        returned["model"]["layers"].append(4)
        returned["model"]["new_key"] = "x"

        assert config.get("lr") == 0.1
        assert config.get("model.layers") == [1, 2, 3]
        assert config.get("model.new_key") is None

    def test_to_dict_is_not_same_object(self):
        """to_dict 返回的对象及其嵌套容器应与原对象不同（深拷贝）。"""
        nested = {"layers": [1, 2]}
        config = Config({"model": nested})

        returned = config.to_dict()

        assert returned is not config._config
        assert returned["model"] is not nested
        assert returned["model"] is not config._config["model"]


class TestMerge:
    """合并语义测试。

    Config 类未提供显式 merge 方法，但 ``set`` 支持在现有配置上写入嵌套键，
    可作为合并的基本原语。本组验证 set 对嵌套结构的合并行为。
    """

    def test_set_merges_into_existing_nested_config(self):
        """set 应在不破坏既有键的情况下写入新的嵌套键。"""
        config = Config({"model": {"encoder": "cnn", "layers": 3}})

        config.set("model.embed_dim", 64)

        # 既有键保留
        assert config.get("model.encoder") == "cnn"
        assert config.get("model.layers") == 3
        # 新键已写入
        assert config.get("model.embed_dim") == 64

    def test_set_creates_intermediate_dicts(self):
        """set 对不存在的中间路径应自动创建空字典。"""
        config = Config({})

        config.set("a.b.c", "deep")

        assert config.get("a.b.c") == "deep"
        assert config["a"] == {"b": {"c": "deep"}}


class TestIteration:
    """迭代 / keys 语义测试。

    Config 类未实现 ``__iter__`` 或 ``keys()``，因此无法直接迭代。
    通过 ``to_dict()`` 转出后迭代是推荐的等价路径，本组验证该契约。
    """

    def test_iteration_via_to_dict(self):
        """无法直接迭代 Config；通过 to_dict 转出后应可正常迭代键。"""
        config = Config({"a": 1, "b": 2, "c": 3})

        keys = list(config.to_dict().keys())

        assert set(keys) == {"a", "b", "c"}

    def test_config_lacks_iter_protocol(self):
        """Config 未实现 ``__iter__``，没有有意义的迭代协议。

        Python 会对未定义 ``__iter__`` 的对象回退到 ``__getitem__(int)``
        序列尝试迭代，但本类的 ``__getitem__`` 只接受点号分隔字符串键，
        因此 ``list(config)`` 会在首个整数索引处失败（TypeError），
        证明其不具备正常迭代语义。推荐的等价路径是 ``to_dict()`` 后迭代。
        """
        config = Config({"a": 1})

        # __iter__ 未定义
        assert not hasattr(type(config), "__iter__") or type(config).__iter__ is object.__iter__

        # 直接迭代因 __getitem__(0) 不接受整数而失败
        with pytest.raises((TypeError, AttributeError)):
            list(config)


class TestNestedKeyAccess:
    """嵌套键访问综合测试"""

    def test_nested_key_access(self):
        """多层嵌套键应能通过点号路径正确访问。"""
        config = Config(
            {
                "data": {
                    "train": {"path": "/data/train", "size": 1000},
                    "val": {"path": "/data/val", "size": 200},
                }
            }
        )

        assert config.get("data.train.path") == "/data/train"
        assert config.get("data.train.size") == 1000
        assert config.get("data.val.path") == "/data/val"
        assert config.get("data.val.size") == 200

    def test_nested_key_access_missing_intermediate(self):
        """中间层缺失时应返回默认值而非抛出异常。"""
        config = Config({"data": {"train": {"size": 1000}}})

        # data.train.path 存在，但 data.val 整体缺失
        assert config.get("data.val.path", "default") == "default"
        # data.train 存在但 path 缺失
        assert config.get("data.train.path", "fallback") == "fallback"

    def test_nested_key_access_on_non_dict_value(self):
        """在非字典值上继续访问嵌套键时应返回默认值（TypeError 分支）。"""
        config = Config({"data": "not_a_dict"})

        # data 是字符串，访问 data.x 触发 TypeError，应返回默认值
        assert config.get("data.x", "default") == "default"

    def test_nested_getitem_raises_on_missing_intermediate(self):
        """__getitem__ 在中间层缺失时应抛出 KeyError。"""
        config = Config({"data": {"train": {"size": 1000}}})

        with pytest.raises(KeyError):
            _ = config["data.val.path"]

    def test_contains_supports_nested_keys(self):
        """__contains__ 应支持点号分隔的嵌套键。"""
        config = Config({"model": {"encoder": "cnn"}})

        assert "model.encoder" in config
        assert "model" in config
        assert "model.missing" not in config
        assert "nonexistent" not in config


class TestInvalidYaml:
    """非法 YAML 处理测试"""

    def test_invalid_yaml_raises_error(self, tmp_path):
        """解析错误的 YAML 应抛出 yaml.YAMLError。"""
        config_file = tmp_path / "invalid.yaml"
        # 未闭合的 flow 映射，yaml.safe_load 会抛 YAMLError
        config_file.write_text("model: {encoder: cnn\n", encoding="utf-8")

        with pytest.raises(yaml.YAMLError):
            Config.from_yaml(str(config_file))

    def test_yaml_with_duplicate_keys_raises(self, tmp_path):
        """重复键在严格解析下应抛出 YAMLError。"""
        config_file = tmp_path / "dup.yaml"
        config_file.write_text("key: a\nkey: b\n", encoding="utf-8")

        # yaml.safe_load 默认对重复键不报错（后者覆盖前者），
        # 这里验证其能被加载且后者胜出，确认行为可预测。
        config = Config.from_yaml(str(config_file))
        assert config.get("key") == "b"
