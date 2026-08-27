"""
chat_api.py
"""

import os
import json
from flask import Blueprint, request, jsonify
import anthropic

chat_bp = Blueprint("chat", __name__)

MODEL_NAME = "claude-sonnet-5"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are the check-in companion inside Wellness Saheli, a women's health app. You act like a warm, skilled therapist and doctor combined -- your job is to actually get to know the user through a real intake conversation before you ever point her at a tool in the app.

CONVERSATION STRUCTURE (do this over many turns, one question at a time):
1. Start broad and emotional: how she's actually doing, her mood lately, energy, sleep.
2. Once she opens up, go deeper the way a therapist intake would -- ask about stress sources: financial stress/money worries, work or study pressure, family conflict, marital or relationship status and how that relationship is going, any relationship or marital problems, loneliness, support system.
3. Ask physical/reproductive questions only after the emotional picture is clear: cycle regularity, pain, symptoms, sexual activity/contraception only if relevant and only after trust is established.
4. Only AFTER you have a real picture across emotional + financial + relational + physical areas should you ever suggest a specific tab. Do not suggest a tab in the first several turns just because one keyword matched.

HOW TO ASK:
- One question at a time, conversational, never a checklist or form-like list.
- Sensitive topics (financial status, marital/relationship problems, family conflict) should be asked gently, with a short reason why you're asking, e.g. "money stress can really affect the body too -- has that been weighing on you?"
- Never assume marital status, relationship status, or sexual activity. Ask, don't assume.

HOW TO RESPOND TO WHAT SHE SHARES:
- Sympathize genuinely and specifically to what she said before asking the next question. Don't just acknowledge-and-pivot.
- When she describes a struggle (financial stress, a difficult marriage, family conflict, loneliness), offer real, concrete coping suggestions for that specific issue -- not generic "take care of yourself" lines.
- Only after a broad enough picture exists, connect what you've learned to a specific tab suggestion with a clear reason tied to what she told you.

STRICT SAFETY RULES:
1. Never diagnose. Never say "you have X." Say things like "some of what you've described is worth checking with the PCOS tool."
2. Never give medical directives (dosages, medication changes). Point to a real provider for that.
3. If she discloses possible self-harm risk, abuse, or an unsafe situation, call record_checkin_insights with crisis_concern true, and stay warm and present in your text reply -- don't go silent, don't try to fix it yourself.
4. Separately from emotional crisis, watch for physical emergency warning signs: extremely severe or sudden pain, fainting or near-fainting, unusually heavy bleeding (soaking through a pad/tampon every hour or less, or passing large clots), symptoms suggesting a possible pregnancy emergency, severe difficulty breathing, or signs of a serious allergic reaction. If any of these come up, call record_checkin_insights with medical_emergency_concern true, and in your text reply calmly and clearly tell her this sounds like something that needs prompt in-person medical attention -- an emergency room or urgent care -- not something to keep discussing in chat. Do not diagnose what it is; just be clear that it needs real, immediate medical attention.
4. Use record_checkin_insights whenever you learn something concrete -- not every turn, only when there's real signal.
5. This should feel like being cared for by someone who actually listens, not screened by a form.
"""

INSIGHTS_TOOL = {
    "name": "record_checkin_insights",
    "description": "Record structured facts learned from the conversation so far, and optionally suggest a specific in-app tool. Only call when there is real signal.",
    "input_schema": {
        "type": "object",
        "properties": {
            "lifestyle": {
                "type": "object",
                "properties": {
                    "regular_exercise": {"type": "boolean"},
                    "exercise_frequency": {"type": "string"},
                    "diet_quality": {"type": "string"},
                    "fast_food_frequent": {"type": "boolean"},
                    "average_sleep_hours": {"type": "number"},
                },
            },
            "mental_health": {
                "type": "object",
                "properties": {
                    "self_reported_stress_level": {"type": "integer"},
                    "notes": {"type": "string"},
                },
            },
            "reproductive_history": {
                "type": "object",
                "properties": {
                    "cycle_regularity": {"type": "string", "enum": ["Regular", "Irregular"]},
                    "cycle_length_days": {"type": "integer"},
                },
            },
            "life_context": {
                "type": "object",
                "description": "Non-medical personal context gathered during the intake.",
                "properties": {
                    "financial_stress": {"type": "boolean"},
                    "relationship_status": {"type": "string"},
                    "relationship_or_marital_difficulty": {"type": "boolean"},
                    "family_conflict": {"type": "boolean"},
                    "support_system_notes": {"type": "string"},
                },
            },
            "suggested_tab": {
                "type": "string",
                "enum": ["pcos", "protection"],
                "description": "Only include once a broad enough picture exists across emotional, relational, financial, and physical areas.",
            },
            "suggested_tab_reason": {"type": "string"},
            "crisis_concern": {"type": "boolean"},
            "medical_emergency_concern": {
                "type": "boolean",
                "description": "True if she described a potential physical medical emergency (severe/sudden pain, fainting, unusually heavy bleeding, possible pregnancy emergency, breathing difficulty, severe allergic reaction).",
            },
        },
    },
}


def _history_to_messages(history):
    messages = []
    for entry in history:
        role = entry.get("role")
        text = entry.get("message", "")
        if role not in ("user", "assistant") or not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


@chat_bp.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True) or {}
    user_message = body.get("message", "").strip()
    history = body.get("history", [])
    profile_context = body.get("profile_context")

    if not user_message:
        return jsonify({"detail": "message is required"}), 422

    try:
        client = _get_client()
    except RuntimeError as e:
        return jsonify({"detail": str(e)}), 500

    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": user_message})

    system_prompt = SYSTEM_PROMPT
    if profile_context:
        system_prompt += (
            "\n\nKNOWN CONTEXT ABOUT THIS USER (use naturally, don't recite it "
            "back like a report):\n" + json.dumps(profile_context, indent=2)
        )

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=800,
            system=system_prompt,
            tools=[INSIGHTS_TOOL],
            messages=messages,
        )
    except Exception as e:
        return jsonify({"detail": f"AI service error: {e}"}), 502

    reply_text = ""
    profile_updates = None
    suggested_tab = None
    suggested_tab_reason = None
    crisis_concern = False
    medical_emergency_concern = False

    for block in response.content:
        if block.type == "text":
            reply_text += block.text
        elif block.type == "tool_use" and block.name == "record_checkin_insights":
            data = block.input or {}
            suggested_tab = data.pop("suggested_tab", None)
            suggested_tab_reason = data.pop("suggested_tab_reason", None)
            crisis_concern = bool(data.pop("crisis_concern", False))
            medical_emergency_concern = bool(data.pop("medical_emergency_concern", False))
            if data:
                profile_updates = data

    return jsonify({
        "reply": reply_text.strip(),
        "profile_updates": profile_updates,
        "suggested_tab": suggested_tab,
        "suggested_tab_reason": suggested_tab_reason,
        "crisis_concern": crisis_concern,
        "medical_emergency_concern": medical_emergency_concern,
    })