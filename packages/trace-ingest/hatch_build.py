# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use repository licensing when building from source; retain it in standalone sdists."""

import email
import shutil
from pathlib import Path
from typing import Any

import tomllib

# Hatch provides this dependency in the isolated build environment.
from hatchling.metadata.plugin import interface  # ty: ignore[unresolved-import]


class RepositoryLicenses(interface.MetadataHookInterface):
    def update(self, metadata: dict[str, Any]) -> None:
        package = Path(self.root)
        repository = package.parents[1]
        if (
            package == repository / "packages" / "trace-ingest"
            and (repository / "pyproject.toml").is_file()
        ):
            project = tomllib.loads((repository / "pyproject.toml").read_text())["project"]
            paths = []
            for pattern in project["license-files"]:
                matches = sorted(path for path in repository.glob(pattern) if path.is_file())
                if not matches:
                    raise FileNotFoundError(f"No repository license files match {pattern!r}")
                for source in matches:
                    relative = source.relative_to(repository)
                    target = package / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    paths.append(relative.as_posix())
        else:
            # PKG-INFO and its license files travel together in the source distribution.
            info = email.message_from_bytes((package / "PKG-INFO").read_bytes())
            paths = info.get_all("License-File", [])
        if not paths or any(not (package / path).is_file() for path in paths):
            raise FileNotFoundError("Source distribution is missing its license files")
        metadata["license-files"] = paths
