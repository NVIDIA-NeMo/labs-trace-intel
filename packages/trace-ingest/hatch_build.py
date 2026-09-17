# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Include the contents of root license links in standalone source archives."""

import os
from typing import Any

from hatchling.builders.hooks.plugin import interface  # ty: ignore[unresolved-import]


class CustomBuildHook(interface.BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        build_data["force_include"] = {
            os.path.realpath(source): target
            for source, target in build_data["force_include"].items()
        }
