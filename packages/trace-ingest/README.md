<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trace ingestion

Trace Intel's existing canonical models and provider loaders, packaged for reuse.
The extraction changes import paths only: schemas, normalization, validation,
ordering, and provider SDK requirements are unchanged.

```python
from trace_ingest import Trace
from trace_ingest.loaders.mlflow import MLflowFileTraceConfig, MLflowFileTraceLoader

loader = MLflowFileTraceLoader(MLflowFileTraceConfig(path="traces.json"))
for trace in loader.load():
    print(trace.id)
```

Use the `mlflow` extra for both MLflow file and live loaders, the `langsmith`
extra for its live loader, and the `langfuse` extra for its live loader.
Consumers own projections into other formats, including ATIF.

## Install

For local development:

```bash
uv add --editable /path/to/labs-trace-intel/packages/trace-ingest --extra mlflow
```

Once a commit containing the package is available remotely:

```bash
uv add "trace-ingest[mlflow] @ git+https://github.com/NVIDIA-NeMo/labs-trace-intel#subdirectory=packages/trace-ingest" --rev <commit-sha>
```

From this repository, `uv sync` installs the workspace package. Existing
`insight_agent.traces` and `insight_agent.trace_loaders` imports remain available
as compatibility exports of the same models and loaders.

## Validate and build

Run the existing loader and application tests with `uv run pytest` from the
repository root. Build both distributions with `uv build --all-packages`.
If publishing distributions, publish `trace-ingest` before `insight-agent`.
