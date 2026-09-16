# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from insight_agent.config import EvidenceStreamsConfig


@pytest.fixture
def select_streams():
    """Explicitly disable other streams in tests exercising a single integration."""

    def select(**selected):
        return dict.fromkeys(EvidenceStreamsConfig.model_fields, False) | selected

    return select
