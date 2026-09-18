# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the two-package release with a local build/registry substitute."""

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "publish_artifactory", Path(__file__).resolve().parents[1] / "tools/publish_artifactory.py"
)
assert spec is not None and spec.loader is not None
publisher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = publisher
spec.loader.exec_module(publisher)


@pytest.mark.parametrize(
    "scenario", ["fresh", "reuse", "conflict", "corrupt", "dry-run", "stable-app"]
)
def test_release_dependency_before_application(tmp_path, monkeypatch, scenario):
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "DIST_DIR", tmp_path)
    monkeypatch.setenv("ARTIFACTORY_PYPI_URL", "https://artifactory.nvidia.com/api/pypi/test")
    monkeypatch.setenv("ARTIFACTORY_TOKEN", "test-secret")
    monkeypatch.setenv("UV_PUBLISH_TOKEN", "other-secret")
    remote = {}
    events = []
    builds = []

    def run(command, *, env):
        if command[1] == "build":
            name = command[command.index("--package") + 1]
            assert "--no-sources" in command and "--wheel" in command
            assert ("--clear" in command) == (not builds)
            assert not {"ARTIFACTORY_TOKEN", "UV_PUBLISH_TOKEN", "UV_PUBLISH_PASSWORD"} & env.keys()
            builds.append(name)
            version = (
                "0.1.0rc1" if name == "insight-agent" and scenario != "stable-app" else "0.1.0"
            )
            stem = f"{name.replace('-', '_')}-{version}"
            path = tmp_path / f"{stem}-py3-none-any.whl"
            with zipfile.ZipFile(path, "w") as wheel:
                wheel.writestr(f"{stem}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
            if name == "trace-ingest" and scenario in {"reuse", "conflict"}:
                remote[path.name] = path.read_bytes() if scenario == "reuse" else b"other release"
        else:
            assert command[1] == "publish"
            assert builds == ["trace-ingest", "insight-agent"]
            assert env["UV_PUBLISH_PASSWORD"] == "test-secret"
            assert "UV_PUBLISH_TOKEN" not in env
            path = Path(command[-1])
            name = publisher._inspect_wheel(path).package_name
            events.append(("dry-run" if "--dry-run" in command else "upload", name))
            if "--dry-run" not in command:
                remote[path.name] = b"corrupt" if scenario == "corrupt" else path.read_bytes()

    def download(url, token):
        name = url.split("/")[-3]
        events.append(("download", name))
        return remote.get(url.rsplit("/", 1)[-1])

    def checksums(url, contents, token):
        events.append(("checksums", url.split("/")[-3]))
        assert contents == remote[url.rsplit("/", 1)[-1]] or scenario == "corrupt"

    monkeypatch.setattr(publisher, "_run", run)
    monkeypatch.setattr(publisher, "_download_artifact", download)
    monkeypatch.setattr(publisher, "_publish_checksums", checksums)

    if scenario == "corrupt":
        with pytest.raises(RuntimeError, match="checksum does not match"):
            publisher.main([])
    else:
        result = publisher.main(["--dry-run"] if scenario == "dry-run" else [])
        assert result == (2 if scenario in {"conflict", "stable-app"} else 0)

    assert builds == ["trace-ingest", "insight-agent"]
    if scenario == "stable-app":
        assert events == []
    elif scenario == "dry-run":
        assert events == [("dry-run", "trace-ingest"), ("dry-run", "insight-agent")]
    elif scenario == "conflict":
        assert events == [("download", "trace-ingest")]
    else:
        expected = [("download", "trace-ingest")]
        if scenario != "reuse":
            expected.append(("upload", "trace-ingest"))
        expected.extend([("checksums", "trace-ingest"), ("download", "trace-ingest")])
        if scenario != "corrupt":
            expected.extend(
                (action, "insight-agent")
                for action in ("download", "upload", "checksums", "download")
            )
        assert events == expected
