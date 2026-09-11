# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify one user message locally."""

from argparse import ArgumentParser
from collections.abc import Sequence

from insight_agent.complaints import load_classifier


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("text", help="One quoted user message")
    parser.add_argument("--device", help="PyTorch device, such as mps or cpu; auto-detected")
    args = parser.parse_args(argv)
    if not args.text.strip():
        parser.error("The user message must not be empty")
    try:
        classifier = load_classifier(args.device)
        result = classifier.classify(args.text)
    except ValueError as exc:
        parser.error(str(exc))
    print(result.model_dump_json(indent=2))
    return 0
