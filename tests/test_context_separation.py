"""One agent working for another gets its own context and hands back a summary."""
import pytest

from agents.hotel_search_agent import HotelSearchAgent
from agents.ops_triage_agent import OpsTriageAgent
from memory import GraphitiMemory, LocalGraphBackend
from runtime import AgentContext, Handover, LLMResponse, LLMToolCall, delegate

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"
USER = "m.saleh"


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.turns: list[list[dict]] = []

    async def complete(self, messages, tools=None):
        self.prompts.append(messages[0]["content"])
        self.turns.append(list(messages))
        return self.replies.pop(0)


def parent_context() -> AgentContext:
    ctx = AgentContext(org_id=ORG, currency="USD", nationality="SA", username=USER,
                       memory=GraphitiMemory(LocalGraphBackend()))
    ctx.remember("hotel_search_session_id", "uuid-parent")
    ctx.memory_context = "What we know: books 5-star only."
    return ctx


# ---- what a child inherits, and what it does not ----

def test_child_gets_the_customer_and_the_durable_store():
    parent = parent_context()
    child = parent.for_child("Check the weather in Jeddah for 1-4 September.")
    assert (child.org_id, child.currency, child.nationality) == (ORG, "USD", "SA")
    assert child.username == USER
    assert child.memory is parent.memory          # durable facts belong to the person
    assert child.brief.startswith("Check the weather")
    assert child.parent is parent


def test_child_cannot_read_the_parent_scratchpad():
    parent = parent_context()
    child = parent.for_child("brief")
    assert child.recall("hotel_search_session_id") is None
    assert child.recall_all() == {}
    assert child.tool_calls == []
    assert child.memory_context is None           # it retrieves for its own question


def test_writing_in_the_child_leaves_the_parent_alone():
    parent = parent_context()
    child = parent.for_child("brief")
    child.remember("selected_option_ref_id", "OPT-9")
    assert parent.recall("selected_option_ref_id") is None
    assert "selected_option_ref_id" not in parent.recall_all()


# ---- delegation returns a summary, not a transcript ----

@pytest.mark.asyncio
async def test_delegate_returns_only_the_answer_and_how_it_was_reached(fake_hasura):
    fake_hasura.responses["Core_BookingQueueStatus"] = [{"Status": "failed"}]
    llm = Scripted(
        LLMResponse(tool_calls=[LLMToolCall("c1", "get_queue_summary", {})]),
        LLMResponse(content="One message failed; recommend escalating to the supplier."))
    parent = parent_context()

    handover = await delegate(OpsTriageAgent(), "Summarise the queue.", llm, parent)

    assert isinstance(handover, Handover)
    assert handover.agent == "ops_triage_agent"
    assert handover.answer.startswith("One message failed")
    assert handover.tools_used == ["get_queue_summary"]
    assert set(handover.to_model()) == {"agent", "answer", "tools_used", "verified", "issues"}
    # nothing the child read is in what came back
    assert "Core_BookingQueueStatus" not in str(handover.to_model())


@pytest.mark.asyncio
async def test_the_parent_context_is_untouched_by_the_delegation(fake_hasura):
    fake_hasura.responses["Core_BookingQueueStatus"] = [{"Status": "failed"}]
    llm = Scripted(
        LLMResponse(tool_calls=[LLMToolCall("c1", "get_queue_summary", {})]),
        LLMResponse(content="Done."))
    parent = parent_context()
    before = dict(parent.recall_all())

    await delegate(OpsTriageAgent(), "Summarise the queue.", llm, parent)

    assert parent.tool_calls == []                # the child's calls are the child's
    assert parent.recall_all() == before
    assert parent.memory_context == "What we know: books 5-star only."


@pytest.mark.asyncio
async def test_the_child_is_told_it_cannot_see_the_conversation(fake_hasura):
    llm = Scripted(LLMResponse(content="Nothing to report."))
    await delegate(OpsTriageAgent(), "Summarise the queue.", llm, parent_context())
    system = llm.prompts[0]
    assert "cannot see its conversation" in system
    assert "one short paragraph" in system


@pytest.mark.asyncio
async def test_the_child_only_ever_receives_the_brief(fake_hasura):
    llm = Scripted(LLMResponse(content="Nothing to report."))
    parent = parent_context()
    parent.remember("private_note", "do not leak this")

    await delegate(OpsTriageAgent(), "Summarise the queue.", llm, parent)

    conversation = str(llm.turns[0])
    assert "Summarise the queue." in conversation
    assert "do not leak this" not in conversation
    assert "uuid-parent" not in conversation


@pytest.mark.asyncio
async def test_a_failing_child_reports_the_problem_rather_than_hiding_it(fake_hasura):
    llm = Scripted(
        LLMResponse(tool_calls=[LLMToolCall("c1", "book_hotel", {"optionRefId": "X"})]),
        LLMResponse(content="Booked."))
    handover = await delegate(OpsTriageAgent(), "Do something forbidden.", llm, parent_context())
    assert handover.passed is False
    assert any("book_hotel" in issue for issue in handover.issues)


@pytest.mark.asyncio
async def test_two_children_do_not_see_each_other(fake_hasura):
    fake_hasura.responses["Core_BookingQueueStatus"] = [{"Status": "failed"}]
    parent = parent_context()
    first = Scripted(LLMResponse(content="First answer."))
    second = Scripted(LLMResponse(content="Second answer."))

    await delegate(OpsTriageAgent(), "First brief.", first, parent)
    await delegate(OpsTriageAgent(), "Second brief.", second, parent)

    assert "First brief." not in str(second.turns[0])


@pytest.mark.asyncio
async def test_a_normal_run_is_not_treated_as_delegated(fake_hasura):
    llm = Scripted(LLMResponse(content="ok"))
    ctx = AgentContext(org_id=ORG, username=USER)
    await HotelSearchAgent().run(ctx, "find me a hotel", llm)
    assert ctx.brief is None
    assert "cannot see its conversation" not in llm.prompts[0]
