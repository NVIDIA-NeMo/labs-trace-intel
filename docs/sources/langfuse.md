<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze Langfuse traces

Choose a time window containing the conversations you want to understand.
Recorded scores, including scores attached to observations, are included automatically.

This adapter supports the v3 API, tested with self-hosted Langfuse 3.205.1 and
Python SDK 3.15. Langfuse v4 is not supported.

## Connect and run

With [uv and Git installed](../../README.md#start-here), install the CLI with Langfuse support:

```bash
uv tool install \
  'insight-agent[langfuse] @ git+https://github.com/NVIDIA-NeMo/labs-trace-intel.git@main'
```

[Configure your inference model and key](../model-access.md#choose-a-model).
Add your project’s Langfuse keys to `.env`:

```dotenv
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
```

Save this as `config.yaml`. Replace the URL and dates with your deployment and trace window:

```yaml
max_tokens: 16384
trace:
  max_traces: 100
  langfuse:
    base_url: https://langfuse.example.com
    from_timestamp: 2026-09-01T00:00:00Z
    to_timestamp: 2026-09-02T00:00:00Z
```

```bash
insight-agent --config config.yaml
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped checks.

## Select the right traces

Your API keys select the project. Both timestamps are required and must include a timezone;
the start is inclusive and the end is exclusive. The default limit is 100 complete traces.
`base_url` overrides `LANGFUSE_BASE_URL` in the environment.

To filter by environment, add this under `trace.langfuse`:

```yaml
filter: '[{"type":"string","column":"environment","operator":"=","value":"production"}]'
```

For tool-schema checks, record tool definitions in `input.tools` on generation observations.
Missing or conflicting catalogs limit those checks.

## Use an export

Replace the `trace` section with a file containing complete v3 trace details:

```yaml
trace:
  max_traces: 100
  langfuse_export:
    path: langfuse-traces.jsonl
```

Rerun the same analysis command. Local loading needs inference credentials only.
Use complete trace-detail records with observations; trace-list summaries and UI CSV files
do not contain enough information. Raw API responses, CLI response envelopes, and directories
of immediate `.json` or `.jsonl` files are accepted. The loader validates all records, rejects
duplicate IDs, and selects the newest traces up to the limit. The extra installs SDK `>=3.15,<4`.

<details>
<summary>Create a complete export</summary>

Set your Langfuse credentials, including `LANGFUSE_BASE_URL`, in `.env` and adjust the dates.
This writes a new file and refuses to overwrite an existing one.

```bash
uv run --isolated --no-project --env-file .env --with 'langfuse>=3.15,<4' python - <<'PY'
from datetime import datetime, timezone
from itertools import count
from langfuse import Langfuse

client = Langfuse(tracing_enabled=False)
with open("langfuse-traces.jsonl", "x", encoding="utf-8") as output:
    for page in count(1):
        result = client.api.trace.list(
            from_timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
            to_timestamp=datetime(2026, 9, 2, tzinfo=timezone.utc),
            page=page, limit=100, order_by="timestamp.asc",
        )
        for trace in result.data:
            output.write(client.api.trace.get(trace.id).json(by_alias=True) + "\n")
        if page >= result.meta.total_pages:
            break
PY
```

</details>
