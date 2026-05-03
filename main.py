"""
Orchestrator -- Main FastAPI Application
=========================================
Exposes endpoints:
  /chat       -- Agentic chat loop with MCP tools and RAG.
  /context    -- Returns current context window usage stats.
  /health     -- Basic health check including backend connectivity.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import (
    LLM_MODEL, LLM_PROVIDER, LLM_API_BASE, MAX_CONTEXT_TOKENS,
    OLLAMA_HOST,
)
from llm_provider import get_provider
from context_manager import context_manager
from context_summarizer import context_summarizer
from conversation_store import conversation_store
from mcp_client import mcp_client
# from rag_service import rag_service  # disabled for MVP — re-enable for RAG

# ---------------------------------------------------------------
# Globals
# ---------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """
Your name is Luna. You were developed by Columbia's Lunar Lions SUITS team.

You are the EVA-team Artificial Intelligence Assistant for the NASA SUITS
2025-2026 mission. Support the EV astronaut during a simulated lunar EVA at the
JSC Rock Yard. Ignore pressurized-rover-only duties unless they directly affect
the EV's location, return path, LTV worksite, live telemetry, or EVA safety.

Mission purpose:
Reduce astronaut cognitive load. Give fast, correct, mission-useful answers in
normal language. Support egress, EV navigation, LTV recovery/repair, return
navigation, ingress, telemetry interpretation, procedure lookup, and map or
equipment understanding.

Response style:
Respond with text only. Do not use markdown, tables, greetings, thanks, or
filler. Be extremely concise, especially for mission-critical data. Prefer short
voice-callout language:
O2 PRI: 94%; O2 SEC: 99%; CO2: 0.05 psi; Action: continue.
For procedures, give the next actionable step first. Use labels like UIA:,
DCU:, HMD:, NAV:, and LTV:. Ask one short clarifying question only when needed.
If a request is unrelated to the EVA mission, politely decline in one sentence.

Voice transcription tolerance:
User messages come from speech-to-text and may contain wrong words, fragments,
or near-homophones. If a request sounds like a garbled version of an EVA task,
telemetry request, procedure name, acronym, location, or piece of equipment,
infer the most likely mission-related meaning from context and proceed. If the
inference would affect safety, switch positions, navigation, or irreversible
procedure steps and confidence is low, ask one short clarification question.
Do not reject plausible mission-related requests just because the wording is
imperfect.

Accuracy and tool use:
Never invent live telemetry, map details, coordinates, switch states, LTV
errors, or procedure text. Use MCP tools when needed and wait for tool results
before answering. Use get_tss_state for live EVA, vitals, UIA, DCU, IMU, LTV,
and LTV error data. Use search_docs/read_doc for mission documents and current
procedures. Use inspect_image for maps, keep-out zones, UIA/DCU photos, or
other visual references. If data is unavailable, say so briefly and state the
best next safe action. Never rely on stale data.

Reference document paths:
Docs root: docs/
Mission description: mission_description/mission-description.pdf
Acronyms: mission_description/suits-acronym-list.pdf
EVA timeline: procedures/ev-team-procedure-timeline.pdf
LTV repair procedures: procedures/ltv-repair-procedures.pdf
EVA coordinates: procedures/ev-team-coordinates.pdf
EVA telemetry ranges: telemetry_ranges/eva-telemetry-ranges.pdf
UIA photo: peripherals/uia.jpeg
DCU photo: peripherals/dcu.jpeg
Rock-yard map: maps/annotated/rock-yard.tiff
Keep-out zones map: maps/annotated/rock-yard-keep-out-zones.tiff
Dust/DUST map: maps/annotated/dust-map.png
When calling document tools, use these paths relative to docs/.

When to retrieve more information:
Use documents or image tools whenever the answer depends on exact procedure
wording, current LTV repair steps, map/keep-out-zone interpretation, unfamiliar
acronyms, coordinates, equipment layout, or a value you are not certain about.
Examples:
If asked for "the NAV restart", "ERM", "fix error 4965", or any LTV repair,
fetch the current procedure before giving steps.
If asked "where do I go", "is this path safe", or "what is that zone", use live
telemetry plus the map/images when available.
If asked "what does ASITS/RIL/POPS mean", answer from known acronyms or verify
in docs if uncertain.
If asked for current vitals, switch states, LTV location, LTV errors, or suit
status, fetch TSS telemetry rather than relying on conversation memory.
If docs/TSS disagree with this prompt, prefer the current retrieved source and
state the conflict briefly only if it matters to the EV.

Operational habits:
When actively guiding a procedure, give one to three steps at a time, then wait
for confirmation or new telemetry unless the user asks for the whole checklist.
Prefer "next step" guidance over dumping long procedures. For switch actions,
include the target panel and state; verify with TSS when available. For
off-nominal telemetry, give value, status, and action. For nominal telemetry,
give only the values the user asked for and "nominal" when useful. Maintain the
current mission phase from conversation context; if phase is unclear, ask a
short phase/location question before giving irreversible procedure steps.

Telemetry alerting note:
A separate hardcoded system handles critical EVA vital alerts. You do not need
to proactively poll or repeatedly alert. When telemetry is requested or relevant
to a task, still note low, high, critical, or off-nominal values and give the
short corrective action from the EVA procedures.

Mission phases for the EVA team:
1. Egress from the airlock using UIA/DCU procedures.
2. Navigate to the assigned LTV Task Board worksite.
3. At the LTV, retrieve and guide Exit Recovery Mode (ERM).
4. Retrieve and guide NAV Restart / Manual Return to Home.
5. Run LTV diagnosis, retrieve current repair procedures from TSS/docs, triage,
   and guide repairs.
6. Navigate back to the PR/HAB using breadcrumbs or best safe path.
7. Complete ingress using UIA/DCU procedures.

Timeline awareness:
The full test session hard stop is 45 min. Ideal EVA flow is about 25 min:
egress 7 min, navigate to LTV 3 min, LTV repair 10 min, return navigation
3 min, ingress 2 min. If time is low, prioritize safety and critical LTV
recovery. Defer non-critical repairs such as dust sensor replacement when needed.

Known EVA coordinates:
Coordinates use TSS meters and may vary by about 10 m during Test Week.
HAB / egress-ingress point: X -5670, Y -10060.
LTV Task Board Alpha: X -5635, Y -9960.
LTV Task Board Bravo: X -5515, Y -9995.
The assigned board is provided during Test Week. Confirm which board before
planning a route when it is unclear.

Navigation behavior:
Help the EV move safely and efficiently under lunar-south-pole-like night
lighting. Use live location, destination, map imagery, keep-out zones, terrain
hazards, POIs, breadcrumbs, and consumables when available. Recommend immediate
alternate routes for hazards. Keep navigation guidance non-obtrusive and
actionable. The docs include rock-yard maps, keep-out-zone overlays, a DUST
coordinate-grid map, and raw 3D terrain meshes. Use image/telemetry tools for
specific route or hazard questions rather than guessing from memory.

Egress procedure summary:
Connect UIA to DCU and start depress:
UIA/DCU: verify EV1 umbilical connection.
UIA: EV1 EMU PWR ON.
DCU: BATT UMB.
UIA: DEPRESS PUMP ON.

Prep O2 tanks:
UIA: OXYGEN O2 VENT OPEN.
HMD: wait until EV1 primary and secondary O2 tanks are less than 10 psi.
UIA: OXYGEN O2 VENT CLOSE.
DCU: OXY PRI.
UIA: OXYGEN EMU-1 OPEN.
HMD: wait until EV1 Primary O2 tank is greater than 2950 psi.
UIA: OXYGEN EMU-1 CLOSE.
DCU: OXY SEC.
UIA: OXYGEN EMU-1 OPEN.
HMD: wait until EV1 Secondary O2 tank is greater than 2950 psi.
UIA: OXYGEN EMU-1 CLOSE.
DCU: OXY PRI.

Prep coolant:
DCU: PUMP OPEN.
UIA: EV1 SUPPLY WATER OPEN.
HMD: wait until EV1 coolant storage is greater than 95%.
UIA: EV1 SUPPLY WATER CLOSE.

End depress, check switches, disconnect:
HMD: wait until suit pressure and O2 pressure are 4 psi.
UIA: DEPRESS PUMP PWR OFF.
DCU: BATT PRI, then BATT LOCAL.
UIA: EV1 EMU PWR OFF.
DCU: FAN PRI.
DCU: PUMP CLOSE.
DCU: CO2 PRI.
DCU: verify OXY PRI.
UIA/DCU: EV1 disconnect umbilical.
Announce egress complete and begin navigation.

Ingress procedure summary:
UIA/DCU: EV1 connect umbilical.
UIA: EV1 EMU PWR ON.
DCU: BATT UMB.
UIA: OXYGEN O2 VENT OPEN.
HMD: wait until primary and secondary O2 tanks are less than 10 psi.
UIA: OXYGEN O2 VENT CLOSE.
DCU: PUMP OPEN.
UIA: EV1 WASTE WATER OPEN.
HMD: wait until EV1 coolant tank is less than 5%.
UIA: EV1 WASTE WATER CLOSE.
UIA: EV1 EMU PWR OFF.
DCU: EV1 disconnect umbilical.

LTV recovery and repair behavior:
Do not hardcode or recite the example LTV repair procedures from memory during
Test Week. The repair PDF is for understanding; current procedures should be
retrieved from TSS/docs when the EV asks or when ltv_errors indicate a repair.
ERM is prerequisite. Confirm the LTV is in recovery mode, usually error 4800,
and that wheels/components are not visibly blocked or damaged. If another error
blocks ERM, fix that first.

LTV error format:
Error codes are 4 digits: criticality digit, subsystem ID, error ID digit,
error ID digit. Criticality runs low-to-high, normally 0-4. Subsystem priority
runs low-to-high, 0-9. The two error ID digits do not indicate priority.
Subsystems:
0 pre-flight/test, 1 secondary computer/HMI, 2 scientific instrumentation,
3 lighting, 4 communications, 5 autonomous guidance/navigation,
6 vehicular mobility/control, 7 thermal regulation,
8 diagnostics/system monitoring, 9 power distribution.
Triage by criticality, subsystem impact, EVA time remaining, and whether the
error blocks recovery. NAV, main power, thermal, mobility/control, and power
distribution faults usually outrank non-critical science or dust-sensor faults.

LTV Task Board abbreviations:
NAV: Navigation. ANAV: Autonomous Navigation. ASITS: Autonomous Systems
Indicators Toggle Switch. RTH: Return to Home. ACA: Autonomy Confidence
Adjustment. PRI: Primary. SEC: Secondary. RIL: Reaction Indicator Light.
POPS: Power Override Panel for Subsystems. RSSI: Received Signal Strength
Indicator.

EVA telemetry nominal ranges and actions:
primary_battery_level and secondary_battery_level: 20-100%.
oxy_pri_storage and oxy_sec_storage: 20-100%.
oxy_pri_pressure and oxy_sec_pressure: 600-3000 psi.
coolant_storage: min 80%, nominal 100%, max 100%.
heart_rate: 50-160 bpm. If high, slow down, control breathing, monitor.
oxy_consumption: 0.05-0.15 psi/min, nominal 0.1.
co2_production: 0.05-0.15 psi/min, nominal 0.1.
suit_pressure_oxy: 3.5-4.1 psi, nominal 4.0. If off-nominal, DCU OXY SEC and
return to PR as soon as possible.
suit_pressure_co2: 0.0-0.1 psi, nominal 0.0. If high, switch DCU CO2 to the
other scrubber.
suit_pressure_other: 0.0-0.5 psi, nominal 0.0. If high after depress, return
to PR as soon as possible.
suit_pressure_total: 3.5-4.5 psi, nominal 4.0. Check O2 pressure and CO2
scrubbers, then follow the relevant action.
helmet_pressure_co2: 0.0-0.15 psi, nominal 0.0. If high, check fan and CO2
state; use DCU FAN SEC for fan-related buildup, switch CO2 if scrubber-related,
and return to PR as soon as possible.
fan_pri_rpm and fan_sec_rpm: expected 30000 rpm when active; min 20000,
max 30000. If the active fan is not at expected RPM, DCU FAN SEC and return to
PR as soon as possible.
scrubber_a_co2_storage and scrubber_b_co2_storage: 0-60%. If beyond 60%,
switch DCU CO2 to the other scrubber to vent the full one.
temperature: 10-32 deg C, nominal 21. If high, slow down, control breathing,
monitor until nominal.
coolant_liquid_pressure: 100-700 psi, nominal 500.
coolant_gas_pressure: 0-700 psi, nominal 0.

Suit system context:
Suit resources are filled during egress and generally decrease during EVA.
Batteries fill when DCU is connected to UIA. Oxygen tanks fill during O2 prep.
Coolant is filled during egress and recycled; coolant dropping below expected
levels can indicate a leak. Fans move helmet CO2 to the scrubbers. CO2 scrubbers
alternate between collecting and venting CO2.

Equipment context:
UIA is the umbilical interface used at the hatch for EV power, O2, water,
depress, and umbilical operations. DCU is the EV suit control unit for BATT
LOCAL/UMB, OXY PRI/SEC, COMMS A/B, FAN PRI/SEC, PUMP OPEN/CLOSE, and CO2
selection. Switch states come from TSS; use telemetry rather than guessing.

Acronyms:
SUITS: Spacesuit User Interface Technologies for Students. EHP: EVA and Human
Surface Mobility Program. TSS: Telemetry Stream Server. EVA: Extravehicular
Activity. EV: astronaut performing the EVA. HMD: Head Mounted Display.
LTV: Lunar Terrain Vehicle. AIA: Artificial Intelligence Assistant.
UIA: Umbilical Interface Assembly. EMU: Extravehicular Mobility Unit.
IMU: Inertial Measurement Unit. DCU: Display and Control Unit.
ERM: Exit Recovery Mode. POI: Point of Interest. DUST: Digital Lunar Exploration
Sites Unreal Simulation Tool.
""".strip()


def _build_system_prompt(tools: list[dict]) -> str:
    """
    Build system prompt that includes a summary of available tools.
    This ensures the model knows what tools exist even when native
    tool calling is not supported by the backend.
    """
    if not tools:
        return _BASE_SYSTEM_PROMPT

    tool_lines = []
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "unknown")
        desc = fn.get("description", "No description")
        params = fn.get("parameters", {}).get("properties", {})
        param_names = ", ".join(params.keys()) if params else "none"
        tool_lines.append(f"  - {name}({param_names}): {desc}")

    tool_block = "\n".join(tool_lines)
    return (
        f"{_BASE_SYSTEM_PROMPT}\n\n"
        f"Available tools:\n{tool_block}\n\n"
        "Do NOT invent or hallucinate tools that are not listed above. "
        "Only reference the tools shown here."
    )

MAX_TOOL_ROUNDS = 10  # Safety valve against infinite tool loops

# LLM provider (chat). Embeddings always stay on Ollama via rag_service.
_llm = get_provider(
    LLM_PROVIDER, ollama_host=OLLAMA_HOST, api_base=LLM_API_BASE,
)

# Per-request context tracking (updated during /chat, queryable via /context)
_last_context_stats: dict = {}


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _coerce_tool_arguments(raw_arguments: Any) -> dict:
    """Normalise tool arguments from Ollama into a JSON object."""
    if raw_arguments is None:
        return {}

    if isinstance(raw_arguments, dict):
        return raw_arguments

    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return {}

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("Tool arguments are not valid JSON.") from exc

        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments JSON must decode to an object.")
        return parsed

    raise ValueError(
        f"Unsupported tool argument type: {type(raw_arguments).__name__}"
    )


def _count_request_tokens(messages: list[dict], tool_schemas: list[dict]) -> int:
    """Count tokens for the message list plus tool schemas sent to Ollama."""
    return context_manager.count_message_tokens(messages, tool_schemas=tool_schemas)


def _normalise_client_messages(messages: list[Any]) -> list[dict]:
    """
    Convert client-provided input into new user messages.

    Conversation history is owned by the server-side JSONL store. The client is
    expected to send only the newest user message, so assistant/tool/system
    messages from clients are ignored to avoid duplicating or spoofing history.
    """
    normalised: list[dict] = []
    for msg in messages:
        content = msg.content
        if msg.role != "user":
            continue
        normalised.append({"role": "user", "content": content})
    return normalised


def _build_assistant_history_message(
    content: str | None,
    tool_calls: list[dict] | None = None,
) -> dict:
    """Format an assistant turn in the message history for the active provider."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content or "",
    }
    if not tool_calls:
        return message

    formatted_tool_calls = []
    for index, call in enumerate(tool_calls):
        tool_call_id = call.get("id") or f"call_{index}"
        function_payload = {
            "name": call["name"],
            "arguments": call["arguments"],
        }

        if LLM_PROVIDER != "ollama":
            function_payload["arguments"] = json.dumps(
                call["arguments"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            formatted_tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": call.get("type", "function"),
                    "function": function_payload,
                }
            )
        else:
            formatted_tool_calls.append({"function": function_payload})

    message["tool_calls"] = formatted_tool_calls
    return message


def _build_tool_history_message(
    content: str,
    tool_call_id: str | None,
    images: list[str] | None = None,
) -> dict:
    """Format a tool result message for the active provider.

    When *images* is provided (list of base64 strings), the images are
    attached so that the LLM can see them directly via Ollama's vision
    support instead of relying on a text-only description.
    """
    message: dict[str, Any] = {
        "role": "tool",
        "content": content,
    }
    if images:
        message["images"] = images
    if LLM_PROVIDER != "ollama" and tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def _tool_message_content_budget(
    messages: list[dict],
    tool_schemas: list[dict],
    tool_message_template: dict | None = None,
) -> int:
    """
    Reserve enough room for the tool message wrapper itself and return
    the remaining content budget for the tool payload.
    """
    current_tokens = _count_request_tokens(messages, tool_schemas)
    remaining_budget = context_manager.get_dynamic_budget(current_tokens)
    empty_tool_message = {"role": "tool", "content": ""}
    if tool_message_template:
        empty_tool_message.update(tool_message_template)
        empty_tool_message["content"] = ""
    empty_tool_tokens = _count_request_tokens(
        messages + [empty_tool_message],
        tool_schemas,
    ) - current_tokens

    if empty_tool_tokens > remaining_budget:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model requested a tool result, but there is no remaining "
                "context budget to fit the tool response."
            ),
        )

    return max(0, remaining_budget - empty_tool_tokens)


# ---------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------

class MessagePayload(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[MessagePayload]
    stream: bool = True


# ---------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    print("[STARTUP] Pre-loading tokenizer …")
    context_manager.load_tokenizer()

    print(f"[STARTUP] Verifying {LLM_PROVIDER} chat backend …")
    llm_health = await _llm.health_check()
    llm_host = llm_health.get("host", OLLAMA_HOST)
    if llm_health.get("reachable", False):
        available = await _llm.list_models()
        if available and any(LLM_MODEL in m for m in available):
            print(
                f"[STARTUP] ✓ Model '{LLM_MODEL}' available on "
                f"{LLM_PROVIDER} at {llm_host}"
            )
        elif available:
            print(
                f"[STARTUP] ⚠ Model '{LLM_MODEL}' NOT found on "
                f"{LLM_PROVIDER}. Available: {available}"
            )
        else:
            print(
                f"[STARTUP] ✓ {LLM_PROVIDER} reachable at {llm_host} "
                f"(model list unavailable or empty)"
            )
    else:
        error = llm_health.get("error", "health check failed")
        print(
            f"[STARTUP] ✗ Cannot reach {LLM_PROVIDER} backend "
            f"at {llm_host}: {error}"
        )

    print("[STARTUP] Connecting to MCP servers …")
    await mcp_client.connect_all()

    print("[STARTUP] Ready.")
    yield

    # ---- Shutdown ----
    print("[SHUTDOWN] Closing MCP connections …")
    await mcp_client.shutdown()


app = FastAPI(
    title="LLM Orchestrator",
    description="Local LLM orchestrator with MCP tools & RAG",
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# /chat endpoint
# ---------------------------------------------------------------

@app.post("/chat")
async def chat(request: ChatRequest):
    """Serialize chat requests through the single shared conversation."""
    async with conversation_store.lock:
        return await _chat_with_history(request)


async def _chat_with_history(request: ChatRequest):
    """
    Main orchestration endpoint.
    Accepts new user messages and returns a (streamed) response.
    """
    global _last_context_stats

    # --- Extract the latest user query ---
    user_query = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_query = msg.content
            break

    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    # --- Gather tool schemas ---
    ollama_tools = mcp_client.get_ollama_tools()

    # --- Load persisted conversation ---
    system_prompt = _build_system_prompt(ollama_tools)
    saved_history = conversation_store.load_history()
    new_user_messages = _normalise_client_messages(request.messages)

    # --- RAG disabled for MVP — re-enable for semantic retrieval ---
    # rag_context, dynamic_budget = await rag_service.retrieve_context(
    #     query=user_query,
    #     budget_limit=dynamic_budget,
    # )
    rag_context = ""
    rag_tokens = 0
    # if rag_context:
    #     rag_tokens = context_manager.count_tokens(rag_context)

    # --- Build the message history ---
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # if rag_context:
    #     messages.append({"role": "system", "content": rag_context})

    messages.extend(saved_history)
    messages.extend(new_user_messages)

    # --- Context summarization when above threshold ---
    pre_summary_tokens = _count_request_tokens(messages, ollama_tools)
    if context_summarizer.should_summarize(pre_summary_tokens):
        messages = await context_summarizer.summarize_history(
            _llm, messages
        )

    pre_tool_tokens = _count_request_tokens(messages, ollama_tools)
    dynamic_budget = context_manager.get_dynamic_budget(pre_tool_tokens)
    if pre_tool_tokens >= MAX_CONTEXT_TOKENS:
        _last_context_stats = {
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "baseline_tokens": max(0, pre_tool_tokens - rag_tokens),
            "rag_tokens": rag_tokens,
            "tool_result_tokens": 0,
            "total_message_tokens": pre_tool_tokens,
            "remaining_budget": 0,
            "utilisation_pct": 100.0,
        }
        raise HTTPException(
            status_code=400,
            detail=(
                "Conversation and retrieved context exceed the model context "
                "limit. Please shorten message history or reduce retrieved context."
            ),
        )

    history_context_messages = messages[1:]

    # --- Track tokens consumed by tool results ---
    tool_result_tokens = 0

    # --- Track tool calls for RAG storage ---
    tool_calls_log: list[dict] = []

    # --- Agentic tool loop ---
    assistant_msg = None
    completed_response = False
    for _round in range(MAX_TOOL_ROUNDS):
        current_tokens = _count_request_tokens(messages, ollama_tools)
        if current_tokens >= MAX_CONTEXT_TOKENS:
            raise HTTPException(
                status_code=400,
                detail="Conversation exceeded the model context limit during tool execution.",
            )

        response = await _llm.chat(
            model=LLM_MODEL,
            messages=messages,
            tools=ollama_tools if ollama_tools else None,
            max_context=MAX_CONTEXT_TOKENS,
        )

        # Strip base64 images from history now that the LLM has seen them.
        # Images are only needed for the single inference call; keeping them
        # wastes ~3-5 K tokens per high-res image on every future turn.
        messages = context_summarizer._strip_images(messages)

        assistant_msg = response

        # No tool calls -> we have a final answer
        if not assistant_msg.tool_calls:
            completed_response = True
            break

        parsed_tool_calls = []
        for call_index, tc in enumerate(assistant_msg.tool_calls):
            tool_name = tc.function.name
            argument_error = None
            tool_call_id = tc.id or f"call_{_round}_{call_index}"
            try:
                tool_args = _coerce_tool_arguments(tc.function.arguments)
            except ValueError as exc:
                argument_error = str(exc)
                tool_args = {}

            parsed_tool_calls.append(
                {
                    "name": tool_name,
                    "id": tool_call_id,
                    "type": tc.type,
                    "arguments": tool_args,
                    "argument_error": argument_error,
                }
            )

        # Append the assistant message (with its tool_calls) to history
        messages.append(
            _build_assistant_history_message(
                assistant_msg.content,
                parsed_tool_calls,
            )
        )

        for call in parsed_tool_calls:
            tool_name = call["name"]
            tool_args = call["arguments"]

            print(f"[TOOL CALL] {tool_name}({json.dumps(tool_args)[:120]})")

            if call["argument_error"]:
                raw_result = f"[TOOL ARGUMENT ERROR] {call['argument_error']}"
            else:
                # Execute via MCP
                raw_result = await mcp_client.execute_tool(tool_name, tool_args)

            # Separate base64 images from text when tool returns vision data
            tool_images: list[str] | None = None
            if isinstance(raw_result, dict):
                tool_images = raw_result.get("images")
                raw_result = raw_result.get("text", "")

            # Log for RAG storage (text only, not base64 blobs)
            tool_calls_log.append({
                "name": tool_name,
                "arguments": tool_args,
                "result": raw_result,
            })

            tool_message_template = {}
            if LLM_PROVIDER != "ollama":
                tool_message_template["tool_call_id"] = call["id"]
            tool_content_budget = _tool_message_content_budget(
                messages,
                ollama_tools,
                tool_message_template=tool_message_template,
            )
            raw_result = context_manager.truncate_to_budget(
                raw_result, tool_content_budget
            )

            result_tokens = context_manager.count_tokens(raw_result)
            tool_result_tokens += result_tokens

            messages.append(
                _build_tool_history_message(raw_result, call["id"], images=tool_images)
            )
            dynamic_budget = context_manager.get_dynamic_budget(
                _count_request_tokens(messages, ollama_tools)
            )

    # --- Compute total context usage for /context endpoint ---
    total_message_tokens = _count_request_tokens(messages, ollama_tools)
    _last_context_stats = {
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "baseline_tokens": max(0, pre_tool_tokens - rag_tokens),
        "rag_tokens": rag_tokens,
        "tool_result_tokens": tool_result_tokens,
        "total_message_tokens": total_message_tokens,
        "remaining_budget": dynamic_budget,
        "utilisation_pct": round(
            (total_message_tokens / MAX_CONTEXT_TOKENS) * 100, 2
        ),
    }

    # --- Build final response ---
    if completed_response and assistant_msg and assistant_msg.content:
        final_content = assistant_msg.content
    elif completed_response:
        final_content = ""
    else:
        final_content = "Unable to complete the request within the tool-call limit."

    # Persist only durable conversation turns. Do not save the system prompt,
    # assistant tool-call scaffolding, or raw tool results; live telemetry should
    # be fetched fresh instead of reused from history.
    updated_history = list(history_context_messages)
    if final_content:
        updated_history.append({"role": "assistant", "content": final_content})
    conversation_store.replace_history(updated_history)

    # --- RAG storage disabled for MVP — re-enable for conversation memory ---
    # if final_content:
    #     asyncio.create_task(
    #         rag_service.store_conversation_turn(
    #             user_message=user_query,
    #             assistant_reply=final_content,
    #             tool_calls_log=tool_calls_log if tool_calls_log else None,
    #         )
    #     )

    # --- Return the response ---
    if request.stream:
        async def generate():
            if final_content:
                yield final_content

        return StreamingResponse(generate(), media_type="text/plain")
    else:
        return {"role": "assistant", "content": final_content}


@app.post("/chat/reset")
async def reset_chat_history():
    """Clear the single server-side conversation history."""
    global _last_context_stats
    async with conversation_store.lock:
        conversation_store.reset_history()
        _last_context_stats = {}
    return {"ok": True}


# ---------------------------------------------------------------
# /context — context window usage
# ---------------------------------------------------------------

@app.get("/context")
async def get_context_usage():
    """
    Returns the token breakdown from the most recent /chat call,
    showing exactly how much of the 128k context window is occupied.
    """
    if not _last_context_stats:
        return {
            "message": "No /chat call has been made yet.",
            "max_context_tokens": MAX_CONTEXT_TOKENS,
        }
    return _last_context_stats


# ---------------------------------------------------------------
# /health — basic health check
# ---------------------------------------------------------------

@app.get("/health")
async def health():
    llm_health = await _llm.health_check()
    llm_reachable = llm_health.get("reachable", False)

    # Check model availability
    model_available = False
    if llm_reachable:
        try:
            models = await _llm.list_models()
            model_available = any(LLM_MODEL in m for m in models)
        except Exception:
            pass

    return {
        "status": "ok" if llm_reachable else "degraded",
        "llm_provider": LLM_PROVIDER,
        "llm_backend": llm_health,
        "model": LLM_MODEL,
        "model_available": model_available,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "mcp_tools_count": len(mcp_client.get_ollama_tools()),
    }


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=13853, reload=False)
