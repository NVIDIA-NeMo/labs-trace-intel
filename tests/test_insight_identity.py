# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from insight_agent.insight import Insight, load_insights


def test_existing_identity_survives_artifact_round_trip(tmp_path):
    insight = Insight(
        id="platform-123", name="Timeouts", description="Repeated timeouts", trace_refs=["a", "b"]
    )
    path = tmp_path / "insights.json"
    path.write_text("[" + insight.model_dump_json() + "]")
    assert load_insights(path)[0].id == "platform-123"


def test_new_insight_does_not_require_or_serialize_an_id():
    insight = Insight(name="Timeouts", description="Repeated timeouts", trace_refs=["a", "b"])
    assert insight.id is None
    assert "id" not in insight.model_dump()


def test_empty_id_is_rejected():
    with pytest.raises(ValidationError):
        Insight(id="", name="Timeouts", description="Repeated timeouts", trace_refs=["a", "b"])
