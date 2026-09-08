#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and validate dependency licenses using the NeMo Platform OSV flow."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "third_party" / "licenses.jsonl"
REQUIREMENTS_PATH = PROJECT_ROOT / "third_party" / "requirements-main.txt"
OSV_PATH = PROJECT_ROOT / "third_party" / "osv-licenses.json"
OVERRIDES_PATH = PROJECT_ROOT / "third_party" / "license_overrides.yaml"
ALLOWED_LICENSES = {
    "APACHE-2.0",
    "BSD-2-CLAUSE",
    "BSD-3-CLAUSE",
    "ISC",
    "MIT",
    "ZLIB",
}


def _canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _scanner() -> str:
    scanner = shutil.which("osv-scanner")
    if scanner is None:
        raise RuntimeError("osv-scanner was not found on PATH")
    return scanner


def _export_requirements(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--all-extras",
            "--quiet",
            "--output-file",
            str(output_path.relative_to(PROJECT_ROOT)),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _scan_licenses(requirements_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    result = subprocess.run(
        [
            _scanner(),
            "scan",
            "source",
            "--licenses",
            "--lockfile",
            str(requirements_path),
            "--format",
            "json",
            "--all-packages",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if not output_path.exists():
        raise RuntimeError(f"OSV-Scanner failed with exit code {result.returncode}")

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    for scan_result in raw.get("results", []):
        scan_result.get("source", {}).pop("path", None)
    output_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _load_overrides() -> dict[str, str]:
    raw = yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
    return {_canonicalize(name): license_name for name, license_name in raw["overrides"].items()}


def _resolve_license(licenses: list[str], override: str | None) -> str | None:
    if override is not None:
        return override.upper()
    for license_name in licenses:
        first_license = re.split(r"\s+(?:AND|OR)\s+", license_name, maxsplit=1)[0].upper()
        if first_license in ALLOWED_LICENSES:
            return license_name.upper()
    return None


def _license_records(osv_path: Path) -> list[dict[str, str | bool]]:
    raw = json.loads(osv_path.read_text(encoding="utf-8"))
    overrides = _load_overrides()
    records: dict[str, dict[str, str | bool]] = {}
    unresolved = []
    used_overrides = set()

    for scan_result in raw.get("results", []):
        for package_data in scan_result.get("packages", []):
            package = package_data["package"]
            name = package["name"]
            key = _canonicalize(name)
            override = overrides.get(key)
            license_name = _resolve_license(package_data.get("licenses", []), override)
            if license_name is None:
                unresolved.append(f"{name}=={package.get('version', '')}")
                continue
            if override is not None:
                used_overrides.add(key)
            records[key] = {
                "name": name,
                "license": license_name,
                "compatible": True,
            }

    if unresolved:
        raise RuntimeError(
            "Packages need reviewed entries in third_party/license_overrides.yaml: "
            + ", ".join(sorted(unresolved, key=str.lower))
        )
    unused_overrides = sorted(set(overrides) - used_overrides)
    if unused_overrides:
        raise RuntimeError("Unused license overrides: " + ", ".join(unused_overrides))
    return [records[key] for key in sorted(records)]


def _render_summary(records: list[dict[str, str | bool]]) -> str:
    return "\n".join(json.dumps(record) for record in records) + "\n"


def _check_file(path: Path, expected: str) -> bool:
    if not path.exists() or path.read_text(encoding="utf-8") != expected:
        print(f"{path.relative_to(PROJECT_ROOT)} is missing or out of date")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of updating files")
    args = parser.parse_args()

    if args.check:
        output_dir = PROJECT_ROOT / "tmp" / "license-check"
        requirements_path = output_dir / REQUIREMENTS_PATH.name
        osv_path = output_dir / OSV_PATH.name
    else:
        requirements_path = REQUIREMENTS_PATH
        osv_path = OSV_PATH

    _export_requirements(requirements_path)
    _scan_licenses(requirements_path, osv_path)
    records = _license_records(osv_path)
    summary = _render_summary(records)

    if args.check:
        valid = _check_file(SUMMARY_PATH, summary)
        if not valid:
            print("Run `make update-licenses` and commit the results.")
        return 0 if valid else 1

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"Wrote {len(records)} dependency license records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
