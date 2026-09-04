from __future__ import annotations

from nooa import Agent

from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult
from insight_agent.insight import Insight
from insight_agent.traces import TraceSnapshot


class InsightCompilation(Agent):
    async def compile_insights(
        self,
        evidence_streams: list[EvidenceStreamResult],
        trace_snapshot: TraceSnapshot,
        existing_insights: list[Insight],
    ) -> list[Insight]:
        """
        Your job is to be the last step of the insight creation process. An
        insight is a bug report or problem that is identified by looking at the
        runtime traces of an AI agent, either in production or an offline
        evaluation context.

        The goal of an insight is to show a developer a problem with their agent
        that they can fix by making a change to their code. Often times this
        could be updating the prompt, changing time outs, fixing code logic,
        updating their agent to use a different model, etc.

        The evidence stream is a set of potential problems that have been
        surfaced by earlier stages. These may be real problems, and they may not
        be fixable.

        Insights should be ranked holistically based on the following factors:
        1. How fixable is this issue by the agent developer?

        2. How high is the  impact of this issue? For example if the formatting
        is slightly incorrect or if there is a contradiction that's not the end
        of the world. If the agent is failing to even produce a response for the
        user 30% of the time though because of an error, that's a huge deal!

        Your job is to validate that these insights are all impactful and
        fixable. If any proposed problem from an Evidence Stream doesn't meet
        this bar, it should be discarded. Some things are not ideal, but are
        recovered by an agent -- for example sometimes an agent will call a
        coding tool that fails, but then go on to recover. There's not anything
        we can do about this issue! And it's OK. Experts do this too. You can
        validate by examining the referenced traces.

        After validating, you must merge the new insights with the existing
        insights, and across evidence streams.

        Here's an example of two insights that are semantic duplicates and
        should be merged:

        1. When users request their airline ticket to be canceled, the system
        prompt allows it but there isn't a tool for it, so the agent tells the
        user that the ticket can't be canceled

        2. Ticket cancelation tool is missing from tool schema

        Here's an example of two insights that should remain independent:

        1. Ticket cancellation calls time out for users that are already checked
        in

        2. Ticket cancellation causes an error when the user has a pending
        refund

        Here's an example of an insight that is overly broad. This kind of
        insight should never be created by you, but if there is an existing one
        in the existing_insights, you should keep it and not modify it.

        1. Bash tool call is returning errors

        Existing insights should never be removed or modified, but you can
        update the trace_refs on existing insights to match new traces you
        identified. Only use trace_refs that come from evidence streams, don't
        worry about searching the trace snapshot to find traces that match a
        particular insight. When you merge insights, also merge the trace_refs.

        Return the list of ranked, validated and merged insights. Include
        existing insights.
        """
        ...


__all__ = ["Insight", "InsightCompilation"]
