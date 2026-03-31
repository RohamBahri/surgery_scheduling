from src.core.config import Config


def test_config_exposes_legacy_and_vfcg_configs() -> None:
    config = Config()

    assert hasattr(config, "legacy_ccg")
    assert hasattr(config, "vfcg")


def test_import_src_vfcg_package() -> None:
    import src.vfcg as vfcg

    assert vfcg is not None
