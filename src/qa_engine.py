"""
qa_engine.py — Person 4: QA Generation, Citations & LLM Integration

Takes the context bundle from Person 3's retrieval system and produces
grounded, cited answers using the Anthropic Claude API.

Responsibilities:
  1. Format retrieved context elements into a structured LLM prompt
  2. Call Claude API and parse back structured JSON (answer + citations + confidence)
  3. Handle multi-turn conversation with memory
  4. Handle unanswerable / ambiguous questions gracefully
  5. Expose a clean QAEngine.ask() interface that Person 5 wires to the demo UI

Dependencies:
  pip install anthropic
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

# import anthropic
# import google.generativeai as genai
from groq import Groq
# ---------------------------------------------------------------------------
# Data shapes that cross the Person 3 → Person 4 boundary
# ---------------------------------------------------------------------------

@dataclass
class ContextElement:
    """
    One element in the context bundle Person 3 hands us.
    Maps directly to the node dict from document_graph.py — we only
    require the fields we actually use in the prompt.
    """
    element_id: str
    type: str           # paragraph / table / image / heading / caption
    content: str        # text content (or generated caption for images)
    page: int
    doc_id: str         # which document this came from
    relevance_score: float = 1.0  # Person 3's retrieval score, used in ranking


@dataclass
class Citation:
    element_id: str
    doc_id: str
    page: int
    type: str
    snippet: str        # short excerpt that directly supports the claim


@dataclass
class QAResult:
    question: str
    answer: str
    citations: list[Citation]
    confidence: float           # 0.0 – 1.0
    is_answerable: bool
    missing_evidence: Optional[str]   # what's lacking when confidence < threshold
    reasoning_path: list[str]         # element_ids traversed to reach the answer
    turn_index: int


# ---------------------------------------------------------------------------
# System prompt (injected once; not repeated per turn)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI Document Analyst embedded in a multi-modal document QA system.
You receive:
  • A numbered list of CONTEXT ELEMENTS — paragraphs, tables, image captions,
    and headings extracted from one or more documents.
  • The user's QUESTION.
  • (Optionally) the CONVERSATION HISTORY so far.

Your job is to produce a JSON response with this exact structure:

{
  "is_answerable": true or false,
  "answer": "your full answer here (empty string if not answerable)",
  "confidence": 0.0 to 1.0,
  "missing_evidence": null or "what information would be needed to answer fully",
  "reasoning_path": ["element_id_1", "element_id_2", ...],
  "citations": [
    {
      "element_id": "...",
      "doc_id": "...",
      "page": 3,
      "type": "paragraph",
      "snippet": "short phrase from the element that supports this claim"
    }
  ]
}

Rules:
- ONLY use information present in the context elements. Never hallucinate.
- If the answer spans multiple elements, cite all of them.
- If no element answers the question, set is_answerable=false, answer="",
  confidence=0.0, and explain in missing_evidence what would be needed.
- confidence reflects how completely the context covers the question:
    1.0  = direct, explicit answer found
    0.7+ = answer can be inferred with reasonable certainty
    0.4+ = partial answer, some gaps
    <0.4 = highly speculative or almost nothing relevant
- reasoning_path lists the element_ids you consulted IN ORDER — this
  powers the visual reasoning path in the UI.
- Keep snippets in citations under 20 words.
- Respond ONLY with the JSON object. No markdown, no preamble.
""").strip()


# ---------------------------------------------------------------------------
# QAEngine
# ---------------------------------------------------------------------------

class QAEngine:
    """
    Main interface for Person 5's demo to call.

    Usage:
        engine = QAEngine()
        result = engine.ask("What drove revenue growth?", context_bundle)
        print(result.answer)
        print(result.citations)

    Multi-turn — just keep calling ask() with updated context:
        result2 = engine.ask("What about APAC?", new_context_bundle)
        # The engine remembers the previous Q&A automatically.
    """

    CONFIDENCE_THRESHOLD = 0.4   # below this → treat as unanswerable

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1500,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        # self.client = anthropic.Anthropic(
        #     api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        # )
        # api = api_key or os.environ.get("GEMINI_API_KEY")
        # genai.configure(api_key=api)
        # self.client = genai.GenerativeModel(
        #     model_name="gemini-1.5-flash",
        #     system_instruction=SYSTEM_PROMPT,
        
        # )
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        # Conversation history — list of {"role": ..., "content": ...}
        self._history: list[dict] = []
        self._turn_index: int = 0

    # -- public API ---------------------------------------------------------

    def ask(
        self,
        question: str,
        context_elements: list[ContextElement],
    ) -> QAResult:
        """
        Ask one question given a list of context elements from Person 3.
        Automatically maintains conversation history for follow-up questions.
        """
        if not context_elements:
            return self._unanswerable(question, "No context elements were retrieved.")

        user_message = self._build_user_message(question, context_elements)
        self._history.append({"role": "user", "content": user_message})

        raw = self._call_llm()

        self._history.append({"role": "assistant", "content": raw})
        self._turn_index += 1

        return self._parse_response(raw, question)

    def reset(self) -> None:
        """Start a fresh conversation (new document session)."""
        self._history = []
        self._turn_index = 0

    def history_summary(self) -> list[dict]:
        """Return Q&A pairs so far — useful for Person 5's conversation panel."""
        pairs = []
        i = 0
        while i + 1 < len(self._history):
            q_msg = self._history[i]
            a_msg = self._history[i + 1]
            if q_msg["role"] == "user" and a_msg["role"] == "assistant":
                # Strip the context block from the stored user message for display
                question_line = q_msg["content"].split("\n\n")[0].replace("QUESTION: ", "")
                pairs.append({"question": question_line, "answer": a_msg["content"]})
            i += 2
        return pairs

    # -- private helpers ----------------------------------------------------

    def _build_user_message(
        self,
        question: str,
        elements: list[ContextElement],
    ) -> str:
        """Serialize context elements into the prompt format the system prompt expects."""
        context_lines = ["CONTEXT ELEMENTS:"]
        for i, el in enumerate(elements, start=1):
            context_lines.append(
                f"\n[{i}] ID: {el.element_id}"
                f"\n    Type: {el.type}"
                f"\n    Document: {el.doc_id}"
                f"\n    Page: {el.page}"
                f"\n    Relevance: {el.relevance_score:.2f}"
                f"\n    Content: {el.content}"
            )

        context_block = "\n".join(context_lines)

        # For follow-up turns we don't re-explain the history —
        # it's already in self._history; just surface the current question.
        if self._turn_index == 0:
            return f"QUESTION: {question}\n\n{context_block}"
        else:
            return (
                f"FOLLOW-UP QUESTION: {question}\n\n"
                f"(Answer using the new context elements below AND the conversation "
                f"history above to understand what the user is referring to.)\n\n"
                f"{context_block}"
            )

    # def _call_llm(self) -> str:
    #     """Send the current history to Claude and return the raw text response."""
    #     response = self.client.messages.create(
    #         model=self.model,
    #         max_tokens=self.max_tokens,
    #         system=SYSTEM_PROMPT,
    #         messages=self._history,
    #     )
    #     return response.content[0].text
    # def _call_llm(self) -> str:
    #     # Convert history to Gemini format
    #     gemini_history = []
    #     for msg in self._history[:-1]:  # all except last
    #         gemini_history.append({
    #             "role": "user" if msg["role"] == "user" else "model",
    #             "parts": [msg["content"]]
    #         })
    
    #     chat = self.client.start_chat(history=gemini_history)
    #     last_msg = self._history[-1]["content"]
    #     response = chat.send_message(last_msg)
    #     return response.text

    def _call_llm(self) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    def _parse_response(self, raw: str, question: str) -> QAResult:
        """Parse Claude's JSON response into a QAResult."""
        try:
            # Strip accidental markdown fences
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: Claude returned prose instead of JSON
            return QAResult(
                question=question,
                answer=raw,
                citations=[],
                confidence=0.5,
                is_answerable=True,
                missing_evidence=None,
                reasoning_path=[],
                turn_index=self._turn_index,
            )

        citations = [
            Citation(
                element_id=c["element_id"],
                doc_id=c["doc_id"],
                page=c["page"],
                type=c["type"],
                snippet=c["snippet"],
            )
            for c in data.get("citations", [])
        ]

        confidence = float(data.get("confidence", 0.0))
        is_answerable = data.get("is_answerable", False) and confidence >= self.CONFIDENCE_THRESHOLD

        return QAResult(
            question=question,
            answer=data.get("answer", "") if is_answerable else "",
            citations=citations,
            confidence=confidence,
            is_answerable=is_answerable,
            missing_evidence=data.get("missing_evidence"),
            reasoning_path=data.get("reasoning_path", []),
            turn_index=self._turn_index,
        )

    def _unanswerable(self, question: str, reason: str) -> QAResult:
        return QAResult(
            question=question,
            answer="",
            citations=[],
            confidence=0.0,
            is_answerable=False,
            missing_evidence=reason,
            reasoning_path=[],
            turn_index=self._turn_index,
        )


# ---------------------------------------------------------------------------
# result_to_dict() — serialization helper for Person 5's API/UI layer
# ---------------------------------------------------------------------------

def result_to_dict(result: QAResult) -> dict:
    return {
        "turn_index": result.turn_index,
        "question": result.question,
        "is_answerable": result.is_answerable,
        "answer": result.answer,
        "confidence": round(result.confidence, 2),
        "confidence_label": _confidence_label(result.confidence),
        "missing_evidence": result.missing_evidence,
        "reasoning_path": result.reasoning_path,
        "citations": [
            {
                "element_id": c.element_id,
                "doc_id": c.doc_id,
                "page": c.page,
                "type": c.type,
                "snippet": c.snippet,
            }
            for c in result.citations
        ],
    }


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "High"
    elif score >= 0.5:
        return "Medium"
    elif score >= 0.4:
        return "Low"
    else:
        return "Insufficient"


# ---------------------------------------------------------------------------
# Self-test (no API key needed — uses mock context like Person 3 would send)
# ---------------------------------------------------------------------------

def _mock_context() -> list[ContextElement]:
    """Simulates what Person 3 returns after graph-traversal retrieval."""
    return [
        ContextElement(
            element_id="doc1_h1",
            type="heading",
            content="Cloud Business",
            page=3,
            doc_id="AnnualReport2025.pdf",
            relevance_score=0.91,
        ),
        ContextElement(
            element_id="doc1_p1",
            type="paragraph",
            content=(
                "Cloud adoption drove most of the revenue increase in FY2025, "
                "contributing approximately 60% of total growth. APAC was the "
                "fastest-growing region, with a 72% year-over-year increase."
            ),
            page=3,
            doc_id="AnnualReport2025.pdf",
            relevance_score=0.97,
        ),
        ContextElement(
            element_id="doc1_t1",
            type="table",
            content=(
                "Revenue by Region: APAC $4.2B (+72%), NA $6.1B (+31%), "
                "EMEA $2.8B (+18%). Total $13.1B (+40% YoY)."
            ),
            page=4,
            doc_id="AnnualReport2025.pdf",
            relevance_score=0.88,
        ),
        ContextElement(
            element_id="doc1_img1",
            type="image",
            content=(
                "A bar chart showing quarterly revenue increasing from $10M in "
                "Q1 2023 to $14M by Q4 2025, with the steepest growth in 2024."
            ),
            page=4,
            doc_id="AnnualReport2025.pdf",
            relevance_score=0.72,
        ),
    ]


if __name__ == "__main__":
    # api_key = os.environ.get("ANTHROPIC_API_KEY")
    # if not api_key:
    #     print("Set ANTHROPIC_API_KEY to run the live test.")
    #     exit(1)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY to run the live test.")
        exit(1)
    
    engine = QAEngine()
    context = _mock_context()

    print("=" * 60)
    print("TURN 1 — Initial question")
    print("=" * 60)
    r1 = engine.ask("Why did revenue increase?", context)
    print(json.dumps(result_to_dict(r1), indent=2))

    print("\n" + "=" * 60)
    print("TURN 2 — Follow-up (should resolve 'APAC' from history)")
    print("=" * 60)
    r2 = engine.ask("What about APAC?", context)
    print(json.dumps(result_to_dict(r2), indent=2))

    print("\n" + "=" * 60)
    print("TURN 3 — Unanswerable question (no relevant context)")
    print("=" * 60)
    empty_context: list[ContextElement] = []
    r3 = engine.ask("What was the CEO's salary?", empty_context)
    print(json.dumps(result_to_dict(r3), indent=2))