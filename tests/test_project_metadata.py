import tomllib
from pathlib import Path


def test_vibevoice_has_no_path_dependency() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "vibevoice" not in metadata.get("tool", {}).get("uv", {}).get("sources", {})


def test_vibevoice_integration_does_not_reference_local_checkout() -> None:
    forbidden_fragments = (
        "../../_ GitHub generell/VibeVoice",
        "vibevoice_repo_path",
        "demo.vibevoice_asr_gradio_demo",
        "sys.path.insert",
        "local VibeVoice checkout",
        "local checkout containing",
    )
    ignored_dirs = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
    ignored_files = {"uv.lock"}

    search_roots = [
        Path("_gh.code-workspace"),
        Path("README.md"),
        Path("NOTES.md"),
        Path("app"),
        Path("config.yaml"),
        Path("pyproject.toml"),
        Path("tests"),
    ]

    matches: list[str] = []
    paths = (
        candidate
        for root in search_roots
        for candidate in ([root] if root.is_file() else root.rglob("*"))
    )
    for path in paths:
        if not path.is_file() or path.name in ignored_files:
            continue
        if path.suffix == ".pyc":
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if ignored_dirs.intersection(path.parts):
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        for fragment in forbidden_fragments:
            if fragment in text:
                matches.append(f"{path}: {fragment}")

    assert matches == []


def test_vibevoice_source_is_vendored() -> None:
    assert Path("vibevoice/modular/modeling_vibevoice_asr.py").is_file()
    assert Path("vibevoice/processor/vibevoice_asr_processor.py").is_file()
    assert Path("app/services/vibevoice_inference.py").is_file()
