"""Conversation loop: drives an Ollama chat model that issues tool calls
against the Intersight MCP server.

The model talks via Ollama's OpenAI-compatible Chat Completions endpoint.
Tool definitions are derived from the live MCP tool list, so adding tools
to the MCP server makes them automatically available to the model with no
Python changes needed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from openai import OpenAI

from mcp_client import IntersightMCPClient, ToolSpec


# Keep models pinned in VRAM between turns so consecutive prompts don't pay
# the model-load cost. Ollama's default is 5 minutes.
KEEP_ALIVE = "24h"

# Sampling temperature for chat completions. We want deterministic tool
# selection — at higher temperatures models (especially Mistral) wander
# between tool calls and prose, sometimes skipping tools the system prompt
# mandates. 0.2 is the well-known sweet spot for instruction-tuned models
# doing function calling: low enough to be reliable, not so low (0.0) that
# we cargo-cult an artifact-prone setting.
TEMPERATURE = 0.2


def _env_int(name: str, default: int) -> int:
    """Parse an integer env var, falling back to `default` on missing/garbage."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(
            f"[orchestrator] WARN: {name}={raw!r} is not an int, "
            f"falling back to {default}",
            file=sys.stderr,
            flush=True,
        )
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Per-request KV-cache size for Ollama. Capping this is the single biggest
# speed knob for big models on a 48 GB GPU: e.g. nemotron-3-super:120b
# advertises 256K max context but allocating that much KV cache forces CPU
# offload and drops throughput ~4x. 8K is plenty for this app's prompts.
NUM_CTX = _env_int("LLM_NUM_CTX", 8192)

# Whether to let models with a "thinking" capability emit visible reasoning
# traces. False ⇒ ~2x faster on nemotron / gpt-oss; ignored by other models.
THINKING_ENABLED = _env_bool("LLM_THINKING", False)


def _build_extra_body() -> dict[str, Any]:
    """Construct the `extra_body` dict for an Ollama chat request.

    Centralized so both the tool-call loop and the format-only path stay in
    sync. `options` is Ollama's per-request runtime-tuning bag (num_ctx,
    num_predict, etc.); `think` is the top-level toggle for reasoning
    models. Models that don't recognize an option silently ignore it, so
    this is safe to send to every model.
    """
    return {
        "keep_alive": KEEP_ALIVE,
        "think": THINKING_ENABLED,
        "options": {
            "num_ctx": NUM_CTX,
        },
    }


def _log(msg: str) -> None:
    """Emit timing diagnostics to stderr; surfaces in `make logs`."""
    print(f"[orchestrator] {msg}", file=sys.stderr, flush=True)


SYSTEM_PROMPT = """\
You are a Cisco Intersight assistant. The user manages compute, network,
and storage infrastructure through Intersight, and you help them inspect
and reason about it.

You have tools that call the Intersight REST API. Follow these rules
exactly — most failures come from breaking them.

TOOL USE
- Always issue tool calls through the function-calling interface. NEVER
  print a tool-call JSON object as text in your reply. If you decide a
  tool is needed, call it; do not narrate it.
- "Announce-and-stop" is the most common failure mode in this app. NEVER
  end a turn with statements like "I will fetch X", "Let me start with
  Y", "Let's list Z first", or "I'll go ahead and call N" without
  actually issuing that tool call in the same turn. The user does not
  see your intent — they see the tool execution. If you decide to call
  a tool, call it right now. If you don't intend to call one, answer the
  question directly.
- For broad inventory questions like "get server details", "show me my
  servers", "what's in my environment", default to calling
  get_physical_servers (unified blade + rack view) right away. Do not
  ask which kind of server first — just fetch and present.
- NEVER ask "what specific information are you looking for", "please
  specify what you need", "I need to know what details", "let me know
  what you want", or any similar CLARIFICATION-STALL phrasing when the
  user's request is reasonable as stated. Broad requests like "list
  servers", "list server info", "show me chassis", "alarm details",
  "list alarms" are valid as-is — pick the right tool and answer them.
  Only ask a clarifying question if the request is genuinely ambiguous
  across UNRELATED resource types (and even then, only one short
  question, as a last resort).
- NEVER respond by listing categories of information you "could"
  provide as a substitute for actually providing it. If the user asks
  "what do you have on servers", you call get_physical_servers and
  SHOW THEM the data — you do NOT enumerate the tools you have access
  to as a menu.
- CONCRETE ROUTING EXAMPLES — follow these patterns exactly:
    "list servers"           -> get_physical_servers
    "list server info"       -> get_physical_servers
    "show me my servers"     -> get_physical_servers
    "server inventory"       -> get_physical_servers
    "what's in my env"       -> get_physical_servers
    "list chassis"           -> get_chassis + get_compute_blades + get_pci_nodes (same turn)
    "chassis info"           -> get_chassis + get_compute_blades + get_pci_nodes (same turn)
    "list chassis details"   -> get_chassis + get_compute_blades + get_pci_nodes (same turn)
    "show alarms"            -> get_alarms
    "list alarms"            -> get_alarms
    "alarm details"          -> get_alarms
    "list alarm details"     -> get_alarms
    "how many alarms"        -> get_alarm_summary
    "any criticals"          -> get_alarm_summary
    "show me fabric"         -> get_fabric_interconnects
    "list FIs"               -> get_fabric_interconnects
    "list profiles"          -> get_server_profiles
    "unassigned profiles"    -> get_server_profiles
    "firmware versions"      -> get_running_firmware
    "HCL status"             -> get_hcl_status
    "contract status"        -> get_contracts
    "advisories"             -> get_advisories
- For ANY chassis-related question ("chassis info", "chassis details",
  "list chassis", "show chassis"), you MUST call ALL THREE of
  get_chassis, get_compute_blades, AND get_pci_nodes before producing
  the final answer. Parallel in one turn is preferred; sequential
  across multiple rounds is acceptable. What is NOT acceptable is
  producing an answer with only one or two of the three called — that
  answer's slot numbers will be WRONG because PCIe nodes occupy chassis
  slots too. If your last tool round called only 1 or 2 of the trio,
  the next round MUST call the missing one(s); only then may you
  format the answer.
- Before producing a final user-visible answer, run a SELF-CHECK
  silently: "What tool calls does the system prompt require for this
  question type? Have I called every one of them?" If the answer is no,
  call the missing tools BEFORE answering. Do not produce an
  intermediate answer and promise to "continue gathering" — gather
  everything first, then answer once.
- The self-check is about MISSING tools, not RE-confirming tools you
  already called. Do NOT call the same tool twice in one user turn
  with the same arguments — once the tool has returned successfully,
  trust the result and use it. Re-calling wastes the user's time.
- Prefer the most specific tool (e.g. get_server_profiles over
  generic_api_call). Use generic_api_call only for endpoints not covered
  by a dedicated tool.
- Chain tool calls when needed: list, then drill in by Moid. Don't ask
  the user for data you can fetch yourself.
- NEVER claim a credentials / authentication / authorization /
  permissions problem unless a tool you ACTUALLY CALLED in this turn
  returned an explicit 401 or 403 with that specific error code in its
  result. Until you have observed such an error in a tool result, your
  job is to call tools, not diagnose them. The user already validated
  credentials before sending the question.
- NEVER claim you "don't have access to the tools needed", "can't do
  this", "lack the ability to", or that a tool "isn't available /
  isn't implemented" without first checking your function-calling
  schema. The schema you are given on every turn is the complete,
  authoritative list of tools you can call — if it lists a tool, you
  have access to it. Apologizing for missing capability is a
  fabrication, same family as the fabricated-auth-error rule above.
  Cross-reference against this app's stable tool list before claiming
  ANY tool is missing:
    list_tools, test_connection, get_server_profiles,
    get_server_profile_by_name, get_physical_servers, get_chassis,
    get_compute_blades, get_compute_rack_units, get_pci_nodes,
    get_fabric_interconnects, get_alarms, get_alarm_summary,
    get_hcl_status, get_running_firmware, get_organizations,
    get_advisories, get_contracts, generic_api_call.
  All 17 of these are available. If the user's request maps to any
  of them (e.g. "list server profiles" -> get_server_profiles), you
  MUST call the tool, not apologize.
- If answering fully requires data from a tool you haven't called yet,
  CALL THAT TOOL. Do not apologize for "missing" data you could fetch.
  Do not stop after one tool call when the question needs several — see
  the chassis rule above for an example.

ODATA QUERY PARAMETERS
- $filter, $top, $skip, $orderby are safe to use freely.
- $select is dangerous: ONLY use field names you have already seen in a
  prior tool result for that resource type. Never invent dotted paths
  like 'DeviceInfo.Sku.Inventory.AvailableSlots'. If you don't know the
  exact field, omit $select entirely — every list tool already returns
  a curated default field set.

INTERSIGHT ERROR SEMANTICS
- Intersight sometimes returns HTTP 403 with `code: "InvalidUrl"` and
  message "Operation not supported. Check if the API path and method
  are valid." This is a REQUEST-VALIDATION error, not a permissions
  problem. It usually means a bad $select field, a wrong path, or a
  wrong method. Retry without $select, or with corrected fields.
- Genuine permission errors come back as 401 or 403 with a different
  code. Only then should you tell the user it's a credentials issue.

DERIVED ANSWERS
- CHASSIS SLOT MATH — this is the #1 silent-failure point of this app.
  Read this entire section before computing any chassis slot numbers.

  PCIe nodes (UCSX-440P, X440p, etc.) occupy chassis slots. A blades-
  only count is WRONG. The formula is:
      used = (blades whose chassis matches) + (PCIe nodes whose paired
              blade is in this chassis)
      free = NumSlots − used

  ALWAYS SHOW YOUR WORK. In the chassis answer you MUST include a line
  for each chassis stating the arithmetic explicitly, e.g.:
      "Used slots = 3 blades + 2 PCIe nodes = 5. Free = 8 − 5 = 3."
  Writing just "Used: 3" without the addition is forbidden — it lets
  the PCIe-skip bug slip through. The explicit "+ N PCIe nodes" forces
  you to actually look at the PCIe node data before producing the
  number.

  WORKED EXAMPLE of the common bug to AVOID. Suppose:
      Chassis A: NumSlots=8
      Blades:    A-blade-1 (slot 1), A-blade-2 (slot 3), A-blade-3 (slot 5)
      PCIe:      A-pcie-1 (slot 2, paired with A-blade-1),
                 A-pcie-2 (slot 4, paired with A-blade-2)
  WRONG answer: "Used: 3, Free: 5"  ← counts only blades
  CORRECT answer: "Used = 3 blades + 2 PCIe nodes = 5. Free = 8 − 5 = 3."

  CHASSIS ANSWER STRUCTURE — the answer to a chassis question is a
  CHASSIS table, not a PCIe table and not a blades table. The PRIMARY
  output is one row per CHASSIS with columns roughly like: Name,
  Model, OperState, NumSlots, Used Slots, Free Slots. The "Used Slots"
  cell MUST contain the arithmetic explicitly, e.g.
  "3 blades + 2 PCIe nodes = 5", not just "5".

  Forbidden output shapes for a chassis question:
    * A table of PCIe nodes with no chassis table.
    * A table of blades with no chassis table.
    * A heading like "Here are the PCIe nodes in your environment"
      (you were asked about chassis, not PCIe nodes).
  PCIe and blade data are INPUTS you use to compute Used Slots; they
  are NOT the user-visible output. You may OPTIONALLY follow the
  chassis summary with a per-slot detail table (one row per occupied
  slot showing whether it's a blade or a PCIe node) if it adds
  context — but only AFTER the chassis summary table, never instead.

  If you produce both a chassis summary AND a per-slot detail table,
  the two must be consistent: if the per-slot table shows 5 occupied
  slots, the summary must report Used=5. Inconsistency between them
  is a bug.

  Field names to use for the join:
    * Blade -> chassis: blade.EquipmentChassis.Moid (canonical) or
      blade.Chassis.Moid (older fallback) — check both.
    * PCIe node -> blade: node.ComputeBlade.Moid (canonical) or
      node.Parent.Moid (fallback) — check both.
  PCIe nodes do NOT reference a chassis directly. Two-hop join:
  pci.Node -> compute.Blade -> equipment.Chassis. Call get_chassis,
  get_compute_blades, AND get_pci_nodes, then do the arithmetic yourself.
- Intersight does NOT always populate `NumSlots` on the chassis MO (often
  empty for X-Series). If NumSlots is 0 or missing, use known capacities
  by model: UCSX-9508 = 8 slots, UCSB-5108-AC2 = 8 slots. If the model is
  not in that list and NumSlots is missing, say so rather than guessing.

PRESENTATION
- ALWAYS reply in English. EVERY character of your reply — including
  greetings, openings, transitions, section titles, table headers,
  bullets, and field labels — must use the standard Latin alphabet
  (a–z, A–Z, 0–9, common punctuation, and emoji where appropriate).
  Do NOT insert characters from non-Latin scripts (Telugu, Devanagari,
  Chinese, Arabic, Cyrillic, etc.) even as stylistic flourishes,
  greetings, or rhetorical openings. If you catch yourself beginning a
  reply with a non-Latin character, restart the reply from scratch.
  This applies regardless of the language the user's question was
  asked in (default to English unless the user explicitly asks for
  another language).
- Use markdown tables for lists. Don't dump raw JSON unless the user
  asks for it.
- EMPTY RESULT HANDLING. When a tool returns an empty result set
  (Results array empty, Count 0, or equivalent), respond with ONE
  clear short factual sentence stating that fact — do NOT produce a
  markdown table at all. An empty table (just a header, or a header
  plus separator with no rows) looks to the user like the reply was
  truncated. Examples:
    Correct: "No active alarms."
    Correct: "No server profiles are currently unassigned."
    Correct: "No advisories affecting this fleet."
    WRONG:   "| Name | Severity | Description"  (header only)
    WRONG:   "| Name | Severity |\n|---|---|"   (header + separator)
  Before writing ANY table, check that the result actually has rows.
  If it doesn't, skip the table entirely and write the short
  statement instead.
- If a query is ambiguous, ask one short clarifying question instead of
  guessing.
- If a tool errors, read the error string carefully — it now includes
  the Intersight error `code` and `message`. Explain plainly and try a
  corrected call when the error suggests one. Never invent data.

Credentials are already configured by the application; you do not need
to call configure_credentials.
"""

# Tools we hide from the model — these are managed by the host app, not the LLM.
HIDDEN_TOOLS = {"configure_credentials"}

# Hard cap on tool-call rounds per user turn. Higher = more headroom for
# legitimate multi-step questions; lower = faster failure on stuck models.
MAX_TOOL_ROUNDS = 12

# If the model issues the SAME tool call with the SAME arguments this many
# times in a row, we cut it off — it's looping, not making progress.
MAX_REPEAT_CALLS = 3


@dataclass
class TurnEvent:
    """Streamed during a turn so the UI can show progress."""

    # "round_start"     — beginning of a new model-call round; UI should clear
    #                     any in-progress assistant text from the previous round
    # "assistant_delta" — incremental text token from a streaming completion
    # "tool_call"       — model issued a tool call
    # "tool_result"     — tool returned (or errored)
    # "assistant_text"  — final assistant text for the turn (authoritative)
    # "error"           — turn aborted with a user-visible error
    kind: str
    name: str = ""
    arguments: dict[str, Any] | None = None
    result_preview: str = ""
    is_error: bool = False
    text: str = ""


@dataclass
class TurnMetrics:
    """Per-turn performance numbers shown in the UI and logs.

    `model_seconds` is the wall time spent inside model_call rounds
    (sum across rounds), distinct from `total_seconds` which also covers
    tool execution and Streamlit overhead. `prompt_tokens` is the LAST
    round's prompt size — that's the peak context usage for the turn,
    after every tool result has been folded back into the messages.
    """

    total_seconds: float = 0.0
    model_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rounds: int = 0
    ctx_max: int | None = None

    @property
    def tok_per_s(self) -> float | None:
        if self.model_seconds > 0 and self.completion_tokens > 0:
            return self.completion_tokens / self.model_seconds
        return None


@dataclass
class TurnRecord:
    """Persisted on the message in chat history so the sidebar can replay it."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    metrics: TurnMetrics = field(default_factory=TurnMetrics)


def mcp_tools_to_openai_schema(tools: Iterable[ToolSpec]) -> list[dict[str, Any]]:
    """Convert MCP ToolSpecs to OpenAI 'function' tool definitions."""
    out: list[dict[str, Any]] = []
    for t in tools:
        if t.name in HIDDEN_TOOLS:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


def _safe_parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


class Orchestrator:
    def __init__(
        self,
        mcp_client: IntersightMCPClient,
        ollama_base_url: str = "http://localhost:11434/v1",
        ollama_api_key: str = "ollama",
    ) -> None:
        self.mcp = mcp_client
        self.client = OpenAI(base_url=ollama_base_url, api_key=ollama_api_key)
        # Stored separately so we can hit Ollama's native /api/show endpoint
        # for context-length metadata. The OpenAI-compat endpoint doesn't
        # expose that.
        self._ollama_base_url = ollama_base_url
        self._ctx_cache: dict[str, int | None] = {}

    def _get_model_context_max(self, model: str) -> int | None:
        """Return the model's max context length, or None on failure.

        Hits Ollama's `/api/show` once per model and caches the result.
        Used for the demo metrics caption in the UI.
        """
        if model in self._ctx_cache:
            return self._ctx_cache[model]
        api_base = self._ollama_base_url.rstrip("/")
        if api_base.endswith("/v1"):
            api_base = api_base[:-3]
        try:
            import httpx  # transitive dep of openai

            with httpx.Client(timeout=5.0) as c:
                r = c.post(f"{api_base}/api/show", json={"name": model})
                r.raise_for_status()
                data = r.json()
            info = data.get("model_info") or {}
            for k, v in info.items():
                if k.endswith(".context_length"):
                    self._ctx_cache[model] = int(v)
                    return self._ctx_cache[model]
        except Exception as exc:
            _log(f"context_length lookup failed for {model}: {exc}")
        self._ctx_cache[model] = None
        return None

    def run_turn(
        self,
        model: str,
        history: list[dict[str, Any]],
        user_message: str,
        on_event: Callable[[TurnEvent], None] | None = None,
    ) -> tuple[str, TurnRecord]:
        """Run one user turn end-to-end.

        Returns (final_assistant_text, TurnRecord). Mutates `history` in place
        with the user message, all assistant tool-call rounds, tool results,
        and the final assistant message — so the next turn picks up the full
        context.
        """
        record = TurnRecord()

        def emit(event: TurnEvent) -> None:
            if on_event is not None:
                on_event(event)

        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        messages = list(history)
        messages.append({"role": "user", "content": user_message})
        history.append({"role": "user", "content": user_message})

        tool_defs = mcp_tools_to_openai_schema(self.mcp.list_tools())
        recent_call_signatures: list[str] = []
        turn_started = time.perf_counter()
        metrics = record.metrics
        metrics.ctx_max = self._get_model_context_max(model)

        def _finalize() -> None:
            metrics.total_seconds = time.perf_counter() - turn_started

        for round_idx in range(MAX_TOOL_ROUNDS):
            emit(TurnEvent(kind="round_start"))

            round_started = time.perf_counter()
            content_buf = ""
            tool_calls_acc: dict[int, dict[str, str]] = {}
            first_token_at: float | None = None

            last_usage = None
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                    tool_choice="auto" if tool_defs else None,
                    temperature=TEMPERATURE,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=_build_extra_body(),
                )
                for chunk in stream:
                    # The usage chunk arrives at the end with an empty `choices`
                    # list, so capture it before the early-continue below.
                    if getattr(chunk, "usage", None):
                        last_usage = chunk.usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if getattr(delta, "content", None):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        content_buf += delta.content
                        emit(TurnEvent(kind="assistant_delta", text=delta.content))

                    for tc_delta in (getattr(delta, "tool_calls", None) or []):
                        idx = tc_delta.index
                        slot = tool_calls_acc.setdefault(
                            idx, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc_delta.id:
                            slot["id"] = tc_delta.id
                        fn = getattr(tc_delta, "function", None)
                        if fn is not None:
                            if fn.name:
                                slot["name"] = fn.name
                            if fn.arguments:
                                slot["arguments"] += fn.arguments
            except Exception as exc:
                msg = f"Ollama call failed: {exc}"
                emit(TurnEvent(kind="error", text=msg, is_error=True))
                history.append({"role": "assistant", "content": msg})
                record.final_text = msg
                _finalize()
                return msg, record

            completion_elapsed = time.perf_counter() - round_started
            ttft = (
                f"{first_token_at - round_started:.2f}s"
                if first_token_at is not None
                else "n/a"
            )
            metrics.model_seconds += completion_elapsed
            metrics.rounds += 1
            if last_usage is not None:
                # prompt_tokens grows each round as tool results are folded
                # back into messages, so the last round is the peak.
                metrics.prompt_tokens = last_usage.prompt_tokens
                metrics.completion_tokens += last_usage.completion_tokens
            _log(
                f"round={round_idx} model_call elapsed={completion_elapsed:.2f}s "
                f"ttft={ttft} content_chars={len(content_buf)} "
                f"tool_calls={len(tool_calls_acc)} "
                f"prompt_tokens={last_usage.prompt_tokens if last_usage else '?'} "
                f"completion_tokens={last_usage.completion_tokens if last_usage else '?'}"
            )

            # No tool calls means the model is done — content_buf is the final answer.
            if not tool_calls_acc:
                emit(TurnEvent(kind="assistant_text", text=content_buf))
                history.append({"role": "assistant", "content": content_buf})
                record.final_text = content_buf
                _finalize()
                rate_str = (
                    f"{metrics.tok_per_s:.1f}"
                    if metrics.tok_per_s is not None
                    else "n/a"
                )
                _log(
                    f"turn complete elapsed={metrics.total_seconds:.2f}s "
                    f"rounds={metrics.rounds} "
                    f"completion_tokens={metrics.completion_tokens} "
                    f"tok_per_s={rate_str}"
                )
                return content_buf, record

            sorted_tcs = [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())]
            assistant_history_entry: dict[str, Any] = {
                "role": "assistant",
                "content": content_buf,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"] or "{}",
                        },
                    }
                    for tc in sorted_tcs
                ],
            }
            messages.append(assistant_history_entry)
            history.append(assistant_history_entry)

            for tc in sorted_tcs:
                tool_name = tc["name"]
                tool_id = tc["id"]
                args = _safe_parse_arguments(tc["arguments"])

                # Loop detection: same tool + same args repeated MAX_REPEAT_CALLS
                # times in a row is the model stuck. Cut it off with a clear note.
                signature = f"{tool_name}::{json.dumps(args, sort_keys=True)}"
                recent_call_signatures.append(signature)
                if len(recent_call_signatures) > MAX_REPEAT_CALLS:
                    recent_call_signatures.pop(0)
                if (
                    len(recent_call_signatures) == MAX_REPEAT_CALLS
                    and len(set(recent_call_signatures)) == 1
                ):
                    msg = (
                        f"The model called `{tool_name}` with the same arguments "
                        f"{MAX_REPEAT_CALLS} times in a row — stopping to avoid a loop. "
                        "Try a different model (e.g. qwen2.5:14b) or a more "
                        "specific question."
                    )
                    emit(TurnEvent(kind="error", text=msg, is_error=True))
                    history.append({"role": "assistant", "content": msg})
                    record.final_text = msg
                    _finalize()
                    return msg, record

                emit(TurnEvent(kind="tool_call", name=tool_name, arguments=args))

                tool_started = time.perf_counter()
                if tool_name in HIDDEN_TOOLS:
                    result_text = json.dumps(
                        {"ok": False, "error": f"Tool {tool_name} is not available to the model."}
                    )
                    is_error = True
                else:
                    try:
                        result = self.mcp.call_tool(tool_name, args)
                        result_text = result.text or json.dumps(
                            {"ok": result.ok, "error": "Empty response"}
                        )
                        is_error = result.is_error
                    except Exception as exc:
                        result_text = json.dumps({"ok": False, "error": str(exc)})
                        is_error = True
                tool_elapsed = time.perf_counter() - tool_started
                _log(
                    f"round={round_idx} tool={tool_name} elapsed={tool_elapsed:.2f}s "
                    f"result_chars={len(result_text)} is_error={is_error}"
                )

                emit(
                    TurnEvent(
                        kind="tool_result",
                        name=tool_name,
                        result_preview=_truncate(result_text, 500),
                        is_error=is_error,
                    )
                )

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_text,
                }
                messages.append(tool_msg)
                history.append(tool_msg)

                record.tool_calls.append(
                    {
                        "name": tool_name,
                        "arguments": args,
                        "result_preview": _truncate(result_text, 1500),
                        "is_error": is_error,
                    }
                )

        msg = (
            "Reached the maximum number of tool-calling rounds for one turn. "
            "Try rephrasing your question or breaking it into smaller steps."
        )
        emit(TurnEvent(kind="error", text=msg, is_error=True))
        history.append({"role": "assistant", "content": msg})
        record.final_text = msg
        _finalize()
        return msg, record

    def run_format_turn(
        self,
        *,
        model: str,
        history: list[dict[str, Any]],
        user_history_message: str,
        format_system_prompt: str,
        format_user_message: str,
        on_event: Callable[[TurnEvent], None] | None = None,
    ) -> tuple[str, TurnRecord]:
        """Single streaming completion with no tools — used by deterministic
        report presets where Python has already gathered the data.

        Two design choices worth flagging:

        * The model call uses a standalone (system, user) message pair built
          from `format_system_prompt` + `format_user_message`. We do NOT mix
          in the existing chat history. That keeps the (potentially huge)
          pre-computed JSON blob out of subsequent turns and lets us swap
          the system prompt to a focused "you are a formatter" without
          mutating chat state.
        * Chat history gets a clean (user, assistant) pair: the short
          `user_history_message` (e.g. "Generate an Intersight inventory
          report.") and the model's formatted reply. So a follow-up turn
          can reference the report by what's visible in the chat without
          drowning in JSON.
        """
        record = TurnRecord()

        def emit(event: TurnEvent) -> None:
            if on_event is not None:
                on_event(event)

        # Keep the chat-history system prompt as-is (or insert the default
        # if this is the very first turn). The format-only call uses its
        # own (system, user) pair below.
        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        history.append({"role": "user", "content": user_history_message})

        messages = [
            {"role": "system", "content": format_system_prompt},
            {"role": "user", "content": format_user_message},
        ]

        metrics = record.metrics
        metrics.ctx_max = self._get_model_context_max(model)
        turn_started = time.perf_counter()

        emit(TurnEvent(kind="round_start"))
        round_started = time.perf_counter()
        content_buf = ""
        first_token_at: float | None = None
        last_usage = None

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=_build_extra_body(),
            )
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    last_usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    content_buf += delta.content
                    emit(TurnEvent(kind="assistant_delta", text=delta.content))
        except Exception as exc:
            msg = f"Ollama call failed: {exc}"
            emit(TurnEvent(kind="error", text=msg, is_error=True))
            history.append({"role": "assistant", "content": msg})
            record.final_text = msg
            metrics.total_seconds = time.perf_counter() - turn_started
            return msg, record

        completion_elapsed = time.perf_counter() - round_started
        ttft = (
            f"{first_token_at - round_started:.2f}s"
            if first_token_at is not None
            else "n/a"
        )
        metrics.model_seconds = completion_elapsed
        metrics.rounds = 1
        if last_usage is not None:
            metrics.prompt_tokens = last_usage.prompt_tokens
            metrics.completion_tokens = last_usage.completion_tokens
        rate_str = (
            f"{metrics.tok_per_s:.1f}"
            if metrics.tok_per_s is not None
            else "n/a"
        )
        _log(
            f"format model_call elapsed={completion_elapsed:.2f}s ttft={ttft} "
            f"content_chars={len(content_buf)} "
            f"prompt_tokens={last_usage.prompt_tokens if last_usage else '?'} "
            f"completion_tokens={last_usage.completion_tokens if last_usage else '?'} "
            f"tok_per_s={rate_str}"
        )

        emit(TurnEvent(kind="assistant_text", text=content_buf))
        history.append({"role": "assistant", "content": content_buf})
        record.final_text = content_buf
        metrics.total_seconds = time.perf_counter() - turn_started
        _log(f"format turn complete elapsed={metrics.total_seconds:.2f}s")
        return content_buf, record
