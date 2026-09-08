#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate source-file copyright and license headers."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NVIDIA_COPYRIGHT = (
    "SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved."
)
SPDX_LICENSE = "SPDX-License-Identifier: Apache-2.0"

HASH_SUFFIXES = {".env", ".py", ".sh", ".toml", ".yaml", ".yml"}
HTML_SUFFIXES = {".md"}
HASH_FILENAMES = {".env.example", ".gitignore", "Makefile"}


def _tracked_and_untracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return [PROJECT_ROOT / name for name in result.stdout.decode().split("\0") if name]


def _comment_style(path: Path) -> str | None:
    if path.name in HASH_FILENAMES or path.suffix in HASH_SUFFIXES:
        return "hash"
    if path.suffix in HTML_SUFFIXES:
        return "html"
    return None


def _nvidia_header(style: str) -> str:
    if style == "hash":
        return f"# {NVIDIA_COPYRIGHT}\n# {SPDX_LICENSE}\n"
    if style == "html":
        return f"<!-- {NVIDIA_COPYRIGHT} -->\n<!-- {SPDX_LICENSE} -->\n"
    raise ValueError(f"Unsupported comment style: {style}")


def _has_nvidia_header(content: str, style: str) -> bool:
    header = _nvidia_header(style)
    if content.startswith("#!"):
        _, separator, remainder = content.partition("\n")
        return bool(separator) and remainder.startswith(header)
    return content.startswith(header)


def _add_nvidia_header(path: Path, content: str, style: str) -> None:
    header = _nvidia_header(style)
    if content.startswith("#!"):
        shebang, separator, remainder = content.partition("\n")
        spacer = "\n" if remainder else ""
        updated = f"{shebang}{separator}{header}{spacer}{remainder}"
    else:
        spacer = "\n" if content else ""
        updated = f"{header}{spacer}{content}"
    path.write_text(updated, encoding="utf-8")


def check_headers(*, fix_nvidia: bool) -> list[Path]:
    invalid: list[Path] = []
    for path in _tracked_and_untracked_files():
        style = _comment_style(path)
        if style is None or not path.is_file():
            continue

        content = path.read_text(encoding="utf-8")
        if _has_nvidia_header(content, style):
            continue
        if fix_nvidia and "SPDX-FileCopyrightText" not in content[:2000]:
            _add_nvidia_header(path, content, style)
            continue
        invalid.append(path.relative_to(PROJECT_ROOT))
    return invalid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix-nvidia",
        action="store_true",
        help="add the exact NVIDIA SPDX header to files known to be NVIDIA-authored",
    )
    args = parser.parse_args()

    invalid = check_headers(fix_nvidia=args.fix_nvidia)
    if invalid:
        print("Files with missing or invalid copyright/license headers:")
        for path in invalid:
            print(f"  {path}")
        print("Run `make update-copyright-headers` only for NVIDIA-authored files.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
