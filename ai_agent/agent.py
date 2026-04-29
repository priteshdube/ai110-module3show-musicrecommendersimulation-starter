"""
agent.py — AI Playlist Planner using Gemini with tool use.

Entry point: plan_playlist(situation, songs)

How the agentic loop works:
  1. User's situation is sent to Gemini as the first message.
  2. Gemini reasons about the situation and emits one or more function_call parts.
  3. We execute each tool call via run_tool() and send the results back as
     function_response parts.
  4. Gemini processes the results, may call more tools, or emits a final text reply.
  5. Loop ends when Gemini stops calling tools (no function_call parts in response).

This pattern is called a "tool-use agentic loop" — Gemini decides when it has
enough information to stop, not the code.
"""

import os
import sys
import json
from typing import List, Dict

from google import genai
from google.genai import types

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_agent.tools import (
    TOOL_DEFINITIONS,
    get_catalog_summary,
    get_recommendations,
    filter_songs_by_attribute,
    get_song_details,
)


# ── System prompt ──────────────────────────────────────────────────────────────
# Sent as the system instruction to Gemini before the user's first message.
# Tells Gemini its role, the catalog structure, and the required output format.

SYSTEM_PROMPT = """You are a smart AI music playlist planner. Your job is to take a \
user's described situation or activity and build them a personalized playlist from a \
catalog of 20 songs.

## Your process — follow this order every time

1. ALWAYS call get_catalog_summary first to see which genres and moods exist in the catalog.
2. Based on the user's situation, reason step by step about:
   - Which genre from the catalog best fits the vibe?
   - Which mood from the catalog best fits?
   - How energetic should the music be? (0.0 = very calm, 1.0 = very intense)
   - Acoustic/organic sound (True) or electronic/produced (False)?
3. Call get_recommendations with those parameters.
4. Optionally call filter_songs_by_attribute or get_song_details to explore further.
5. Write your final response to the user.

## Song catalog field reference

- genre / mood: categorical — only use values from get_catalog_summary
- energy: 0.0 (very calm, background music) → 1.0 (very intense, workout music)
- acousticness: 0.0 (fully electronic) → 1.0 (fully acoustic/organic)
- valence: 0.0 (dark/sad) → 1.0 (bright/happy)
- danceability: 0.0 (not danceable) → 1.0 (highly danceable)
- tempo_bpm: raw beats per minute

## Final response format

Write a short friendly intro explaining your reasoning (1–2 sentences), \
then present the playlist like this:

**Your Playlist:**
1. **[Song Title]** by [Artist] — [one sentence on why it fits the situation]
2. ...

Keep the tone conversational and make the listener feel like you understood them.
"""


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def run_tool(tool_name: str, args: Dict, songs: List[Dict]) -> str:
    """
    Receives a tool call from Gemini (name + args dict) and dispatches it
    to the correct Python function in tools.py.

    Returns a JSON string — Gemini expects tool results as serializable data.
    Errors are caught and returned as JSON so the loop never crashes mid-run.
    """
    try:
        if tool_name == "get_catalog_summary":
            result = get_catalog_summary(songs)

        elif tool_name == "get_recommendations":
            result = get_recommendations(songs, **args)

        elif tool_name == "filter_songs_by_attribute":
            result = filter_songs_by_attribute(songs, **args)

        elif tool_name == "get_song_details":
            result = get_song_details(songs, **args)

        else:
            result = {"error": f"Unknown tool: '{tool_name}'"}

        return json.dumps(result)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Agentic loop ───────────────────────────────────────────────────────────────

def plan_playlist(situation: str, songs: List[Dict], max_turns: int = 10) -> str:
    """
    Main entry point for the AI playlist planner.

    Args:
        situation:  Natural-language description of what the user is doing/feeling.
        songs:      The loaded song catalog (list of dicts from load_songs()).
        max_turns:  Safety cap on tool-use rounds to prevent runaway API calls.

    Returns:
        A formatted string with Gemini's final playlist and explanation.

    Raises:
        ValueError: If GOOGLE_API_KEY is not set in the environment.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Add it to your .env file or export it in your shell."
        )

    # Client reads GEMINI_API_KEY from the environment automatically
    client = genai.Client()

    chat = client.chats.create(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[TOOL_DEFINITIONS],
        ),
    )

    # --- Turn 0: send the user's situation ---
    response = chat.send_message(situation)

    for _ in range(max_turns):
        # Collect every function_call part from this response
        fn_calls = [
            part.function_call
            for part in response.parts
            if part.function_call and part.function_call.name
        ]

        # No function calls → Gemini is done; return its text response
        if not fn_calls:
            return response.text or ""

        # Execute each tool call and build function_response parts to send back
        tool_result_parts = []
        for fc in fn_calls:
            result_str = run_tool(fc.name, dict(fc.args), songs)
            tool_result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"output": result_str},
                    )
                )
            )

        # Send all tool results back to Gemini in a single message
        response = chat.send_message(tool_result_parts)

    # Safety fallback: return whatever text is in the last response
    return response.text or "Could not generate a playlist. Please try again."
