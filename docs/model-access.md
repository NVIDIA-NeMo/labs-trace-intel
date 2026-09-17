<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data and model access

Trace Analyst needs an inference model to investigate traces and write insights.
Its API key is separate from your Braintrust, LangSmith, Langfuse, MLflow, or Intake credentials.

## Choose a model

Save your model settings in `.env`. For OpenAI, use:

```dotenv
INSIGHT_AGENT_MODEL=openai/gpt-5.2
INSIGHT_AGENT_API_KEY=your-inference-key
```

For an approved OpenAI-compatible gateway, use its model name and API base:

```dotenv
INSIGHT_AGENT_MODEL=openai/your-model-name
INSIGHT_AGENT_API_KEY=your-gateway-key
INSIGHT_AGENT_API_BASE=https://gateway.example.com/v1
```

Replace the example values with your provider’s settings. Use the documented API base,
without appending a resource path such as `/chat/completions`.
The first `openai/` selects the protocol; the rest is the gateway’s model name.

The source guides set `max_tokens: 16384` to limit each model response.
Adjust that YAML setting to a value your endpoint supports.
You can also set `model` and `api_base` in YAML or override them on the CLI.

## Credentials

Keep `.env` in the folder where you run the command to load it automatically:

```bash
insight-agent --config config.yaml
```

Or select an environment file explicitly with uv:

```bash
uv tool run --env-file /path/to/.env insight-agent --config config.yaml
```

Exported variables take priority over values in either file.
Keep credentials out of configuration YAML and source control.

`OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are accepted as fallback inference keys.
`OPENAI_API_BASE` and `OPENAI_BASE_URL` are fallback endpoint settings.
Choose the matching model explicitly. The CLI requires an inference key before loading traces.

Each source guide names the credentials needed for live access.
Loading a saved export does not require credentials for that trace platform.

Return to your [source guide](../README.md#start-here) to finish setup.

## Where data goes

| When you use… | Data access |
| --- | --- |
| A live trace source | Trace Analyst reads traces and recorded evaluation signals from that service. |
| Model-based analysis | Trace content, evaluation signals, and candidate findings may be sent to your configured inference provider. |
| An ethos document | Its contents are included in model-based analysis. |
| Code validation | Selected source excerpts are sent to the inference provider. The validator has read-only file tools. |
| Remote sentiment embeddings | Extracted user messages are sent to the configured embedding endpoint. |
| Local sentiment embeddings | The embedding model runs on your machine; model and tokenizer files may download on first use. |

Remote sentiment also loads a tokenizer, which may need an initial download.
Using local trace files still requires inference access. There is no automatic redaction
of trace content before analysis; choose inputs and endpoints appropriate for your data.

The CLI writes insights to a local YAML file or stdout. It does not publish findings
back to the trace platform. [Output details](results.md#saved-output).
