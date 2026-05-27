import tomllib
from pathlib import Path


def test_vibevoice_is_not_required_for_base_install() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "vibevoice" not in metadata["project"]["dependencies"]
    assert "vibevoice" not in metadata.get("tool", {}).get("uv", {}).get("sources", {})
