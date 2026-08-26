"""Keep the bundled skill honest.

A skill that documents fields the schema no longer has, or misses rules that
were added, is worse than no skill: an agent will follow it confidently into
the wrong shape. These tests cannot check prose accuracy, but they can check
that every field name and every finding type is at least mentioned.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from insight_agent.coverage import RULE_REQUIREMENTS
from insight_agent.evidence_streams.tool_issues import FINDING_TYPES
from insight_agent.validate import trace_schema
from insight_agent.venue import VenueProfile

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "insight-trace-adapter"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCE = SKILL_DIR / "reference"


def schema_property_names() -> set[str]:
    schema = trace_schema()
    names = set(schema["properties"])
    for definition in schema["$defs"].values():
        names |= set(definition["properties"])
    return names


def test_skill_exists_with_required_frontmatter():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n")

    frontmatter = text.split("---", 2)[1]
    assert re.search(r"^name:\s*insight-trace-adapter\s*$", frontmatter, re.M)
    description = re.search(r"^description:\s*(.+)$", frontmatter, re.M)
    assert description and len(description.group(1)) > 80


def test_description_carries_literal_trigger_phrases():
    """The description is the only thing in context when deciding to load."""
    frontmatter = SKILL_MD.read_text(encoding="utf-8").split("---", 2)[1]
    for phrase in ('"write an adapter"', '"convert my traces"', '"onboard a new'):
        assert phrase in frontmatter, phrase


def test_every_bundled_file_referenced_by_the_skill_exists():
    text = SKILL_MD.read_text(encoding="utf-8")
    for link in re.findall(r"\]\((?!https?:)([^)#]+)\)", text):
        if link.startswith("../"):
            continue  # repo-relative links are checked separately
        assert (SKILL_DIR / link).exists(), link


@pytest.mark.parametrize("issue_type", sorted(FINDING_TYPES))
def test_every_finding_type_appears_in_the_capability_matrix(issue_type):
    text = (REFERENCE / "capability-matrix.md").read_text(encoding="utf-8")
    assert issue_type in text, (
        f"{issue_type} is missing from the capability matrix. Regenerate it so adapter "
        "authors can see what gates it."
    )


@pytest.mark.parametrize("name", sorted(schema_property_names()))
def test_every_schema_field_appears_in_the_canonical_schema_reference(name):
    text = (REFERENCE / "canonical-schema.md").read_text(encoding="utf-8")
    assert f"`{name}`" in text, (
        f"schema field {name!r} is undocumented in the skill's reference. Regenerate "
        "docs/canonical-schema.md and copy it into the skill."
    )


def test_capability_matrix_covers_exactly_the_known_rules():
    assert {r.issue_type for r in RULE_REQUIREMENTS} == set(FINDING_TYPES)


def test_the_tool_catalog_uplift_claim_is_the_measured_number():
    """The skill tells people `tool_catalog` unlocks N rules. Measure N.

    An earlier draft claimed 7 (counting the schema half of malformed_tool_call,
    which fires with or without a catalog). Adapter authors prioritise work off
    this number, so it is pinned here rather than left to prose.
    """
    import json
    import warnings

    from insight_agent.coverage import corpus_coverage
    from insight_agent.trace_loaders import InsightTraceV1Loader

    corpus_path = REPO_ROOT / "src" / "insight_agent" / "data" / "sample_corpus.jsonl"
    records = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stripped = [{k: v for k, v in r.items() if k != "tool_catalog"} for r in records]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = corpus_coverage(InsightTraceV1Loader.from_records(records))
        without = corpus_coverage(InsightTraceV1Loader.from_records(stripped))

    uplift = full["rules"]["evaluable"] - without["rules"]["evaluable"]
    gated = sum(1 for r in RULE_REQUIREMENTS if r.needs == "tool_catalog")

    assert uplift == gated == 6
    assert f"**{uplift} rules at once**" in SKILL_MD.read_text(encoding="utf-8")


def test_skill_reference_matches_the_repo_documentation():
    """The skill's schema reference is a copy; it must not drift."""
    assert (REFERENCE / "canonical-schema.md").read_text(encoding="utf-8") == (
        REPO_ROOT / "docs" / "canonical-schema.md"
    ).read_text(encoding="utf-8")


def test_the_seven_traps_are_all_documented():
    text = (REFERENCE / "traps.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \d+\.", text, re.M)
    assert len(headings) == 7

    for marker in (
        "result_missing",
        "explicit_error",
        "content",
        "result_id",
        "call_id",
        "arguments",
        "complete_provenance_context",
        "logical_case_id",
    ):
        assert marker in text, marker


def test_skill_documents_the_required_fields_exactly():
    text = SKILL_MD.read_text(encoding="utf-8")
    schema = trace_schema()
    for name in schema["required"] + schema["$defs"]["call"]["required"]:
        assert name in text, name


def test_skill_names_the_verify_loop_commands():
    text = SKILL_MD.read_text(encoding="utf-8")
    for command in (
        "insight-agent schema",
        "insight-agent validate",
        "insight-agent coverage",
        "insight-agent explain-failures",
        "insight-agent init-adapter",
    ):
        assert command in text, command


def test_bundled_template_is_valid_python_and_teaches_the_loop():
    source = (SKILL_DIR / "assets" / "adapter_template.py").read_text(encoding="utf-8")
    compile(source, "adapter_template.py", "exec")
    assert "insight-agent validate" in source
    assert "missing_tool_result" in source


def test_venue_profile_fields_are_documented():
    text = (REPO_ROOT / "docs" / "venue-profiles.md").read_text(encoding="utf-8")
    for field in VenueProfile().to_dict():
        assert f"`{field}`" in text, field


def test_readme_links_resolve():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for link in re.findall(r"\]\((?!https?:)([^)#]+)\)", text):
        assert (REPO_ROOT / link).exists(), link
