<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trace ingestion

Canonical trace models and provider loaders for use outside Trace Analyst.

Install a pinned Git revision:

```bash
uv add "git+https://github.com/NVIDIA-NeMo/labs-trace-intel#subdirectory=packages/trace-ingest" --rev <commit-sha> --extra mlflow
```

Load an MLflow export:

```python
from trace_ingest.loaders.mlflow import MLflowFileTraceConfig, MLflowFileTraceLoader

loader = MLflowFileTraceLoader(MLflowFileTraceConfig(path="traces.json"))
for trace in loader.load():
    print(trace.id)
```

Use the `mlflow` extra for both MLflow file and live loaders, the `langsmith`
extra for its live loader, and the `langfuse` extra for its live loader.

For local development, use an editable checkout:

```bash
uv add --editable /path/to/labs-trace-intel/packages/trace-ingest --extra mlflow
```
