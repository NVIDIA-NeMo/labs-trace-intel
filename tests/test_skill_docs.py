"""Keep the bundled loader-authoring skill tied to executable contracts."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "trace-loader"
SKILL_MD = SKILL_DIR / "SKILL.md"


def test_skill_exists_with_required_frontmatter():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n")

    frontmatter = text.split("---", 2)[1]
    assert re.search(r"^name:\s*trace-loader\s*$", frontmatter, re.M)
    description = re.search(r"^description:\s*(.+)$", frontmatter, re.M)
    assert description and "TraceLoader" in description.group(1)


def test_skill_points_to_live_loader_contracts():
    text = SKILL_MD.read_text(encoding="utf-8")
    paths = (
        "src/insight_agent/trace_loaders/trace_loaders.py",
        "src/insight_agent/trace_loaders/mlflow.py",
        "src/insight_agent/trace_loaders/fs.py",
        "src/insight_agent/traces.py",
    )
    for path in paths:
        assert path in text
        assert (REPO_ROOT / path).exists()


def test_skill_has_no_guessed_source_template_or_copied_schema():
    bundled_files = {
        path.relative_to(SKILL_DIR).as_posix() for path in SKILL_DIR.rglob("*") if path.is_file()
    }
    assert bundled_files == {"SKILL.md"}

    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Do not guess fields" in text
    assert "sole canonical definition" in text


def test_readme_links_resolve():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for link in re.findall(r"\]\((?!https?:)([^)#]+)\)", text):
        assert (REPO_ROOT / link).exists(), link
