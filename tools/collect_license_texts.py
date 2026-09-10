# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect notices from checksum-verified locked archives without installing packages."""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import TypedDict
from urllib.request import urlopen


class Artifact(TypedDict):
    url: str
    hash: str


class Document(TypedDict):
    source: str
    sha256: str
    path: str
    text: str


def _is_notice(path: str) -> bool:
    name = PurePosixPath(path).name
    return bool(
        re.fullmatch(
            r"(?:.*[-_])?(?:licen[cs]es?|copying|notices?|copyright|authors|acknowledg[e]?ments)"
            r"(?:[._-][\w.-]+)?",
            name,
            flags=re.IGNORECASE,
        )
    ) and not name.lower().endswith((".py", ".pyc", ".json", ".html", ".c", ".h", ".rtf", ".pdf"))


def _archive_documents(data: bytes, filename: str) -> dict[str, bytes]:
    """Read license files and declared License-File entries, never extract to disk."""
    if filename.endswith((".whl", ".zip")):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
            selected = {name for name in names if _is_notice(name)}
            for name in names:
                if name.endswith((".dist-info/METADATA", "/PKG-INFO")):
                    metadata = BytesParser().parsebytes(archive.read(name), headersonly=True)
                    for declared in metadata.get_all("License-File", []):
                        matches = {p for p in names if p == declared or p.endswith("/" + declared)}
                        if not matches:
                            raise RuntimeError(
                                f"{filename}: declared license file missing: {declared}"
                            )
                        selected.update(matches)
            return {name: archive.read(name) for name in sorted(selected)}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        members = {item.name: item for item in archive.getmembers() if not item.isdir()}
        selected = {name for name in members if _is_notice(name)}
        for name, member in members.items():
            if name.endswith("/PKG-INFO") and member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                metadata = BytesParser().parsebytes(stream.read(), headersonly=True)
                for declared in metadata.get_all("License-File", []):
                    matches = {p for p in members if p.endswith("/" + declared)}
                    if not matches:
                        raise RuntimeError(f"{filename}: declared license file missing: {declared}")
                    selected.update(matches)
        documents = {}
        for name in sorted(selected):
            member = members[name]
            if not (member.isfile() or member.issym() or member.islnk()):
                raise RuntimeError(f"{filename}: unsupported license file type: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"{filename}: unreadable license file: {name}")
            documents[name] = stream.read()
        return documents


def _download(artifact: Artifact, cache: Path) -> bytes:
    digest = artifact["hash"].removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or not artifact["hash"].startswith("sha256:"):
        raise RuntimeError(f"Expected SHA256 checksum for {artifact['url']}")
    if not artifact["url"].startswith("https://"):
        raise RuntimeError(f"Expected HTTPS archive URL: {artifact['url']}")
    path = cache / digest
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(artifact["url"], timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f"Archive checksum mismatch: {artifact['url']}")
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def collect(
    package: dict, cache: Path, supplements: list[Artifact] | None = None
) -> list[Document]:
    """Prefer universal wheels, then source archives, then a deterministic wheel fallback.

    Artifact selection is independent of the machine running the collector. Binary
    wheels for arbitrary deployment platforms are not a substitute for auditing a
    redistributed environment; this inventory documents this source/Python package.
    """
    wheels = sorted(package.get("wheels", []), key=lambda item: item["url"])
    universal = [w for w in wheels if w["url"].endswith("-none-any.whl")]
    candidates = universal[:1]
    if package.get("sdist"):
        candidates.append(package["sdist"])
    if not universal and wheels:
        candidates.append(wheels[0])
    documents: list[Document] = []
    for artifact in candidates:
        data = _download(artifact, cache)
        files = _archive_documents(data, artifact["url"])
        if not any(re.search(r"licen[cs]e|copying", PurePosixPath(p).name, re.I) for p in files):
            continue
        for path, content in files.items():
            text = content.decode("utf-8")
            if "\0" in text:
                raise RuntimeError(f"Non-text license/notice file in {artifact['url']}: {path}")
            if not text.strip():
                raise RuntimeError(f"Empty license/notice file in {artifact['url']}: {path}")
            documents.append(
                {"source": artifact["url"], "sha256": artifact["hash"], "path": path, "text": text}
            )
        break
    for artifact in supplements or []:
        text = _download(artifact, cache).decode("utf-8")
        if not text.strip():
            raise RuntimeError(f"Empty supplemental license: {artifact['url']}")
        documents.append(
            {
                "source": artifact["url"],
                "sha256": artifact["hash"],
                "path": "supplemental license",
                "text": text,
            }
        )
    if not documents:
        raise RuntimeError(
            f"No license text in locked distributions for {package['name']}=={package['version']}. "
            "Add a reviewed exception in third_party/license_exceptions.yaml."
        )
    return documents


def render_texts(packages: list[tuple[str, str, list[Document]]]) -> str:
    lines = [
        "Third-party license and attribution texts",
        "Generated by make update-licenses from locked distributions.",
        "Upstream texts retain their own licenses and copyright notices.",
        "Identical documents within each package are printed once with all source paths.",
        "",
    ]
    for name, version, documents in packages:
        lines.extend(["=" * 80, f"{name}=={version}", "=" * 80, ""])
        groups: dict[str, list[Document]] = {}
        for document in documents:
            groups.setdefault(document["text"], []).append(document)
        for content, originals in groups.items():
            for original in originals:
                lines.extend(
                    [
                        f"Source: {original['source']}",
                        f"SHA256: {original['sha256']}",
                        f"File: {original['path']}",
                    ]
                )
            lines.extend(["", content, ""])
    return "\n".join(lines) + "\n"
