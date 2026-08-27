"""
chat.py

Implements POST /chat, used by the AI Check-In tab
(lib/services/ai_checkin_service.dart -> ChatTurnResult).

Design:
  - This calls the Anthropic API (Claude) IF an API key is configured
    via the ANTHROPIC_API_KEY environment variable. You (the project
    owner) need to set this yourself -- it is intentionally NOT
    hardcoded anywhere in this file or committed to git. See the
    README section this was added alongside for how to set it on
    PythonAnywhere.
  - If no key is configured, /chat still works (so the rest of the
    app -- and your demo -- doesn't break) but returns a clearly
    labelled canned response instead of silently pretending to be an
    AI. This matters for your viva: an examiner asking "what happens
    if the AI service is unavailable" has a real, honest answer
    instead of a crash.
  - MentalHealthFlags / privacy: the health_profile.dart docstring is
    explicit that this app "must never claim to diagnose anxiety,
    depression, or any mental health condition." The system prompt
    below enforces that at the model level, and the lightweight
    keyword-based crisis check below is a SEPARATE, deterministic
    safety net that doesn't rely on the model behaving -- if it fires,
    the reply is replaced with real crisis resources regardless of
    what the model said.
"""

import os

CRISIS_KEYWORDS = [
    "kill myself", "suicide", "end my life", "want to die",
    "hurting myself", "hurt myself", "self harm", "self-harm",
    "no reason to live", "better off dead",
]

CRISIS_REPLY = (
    "I'm really glad you told me this, and I want to make sure you get "
    "real support right now, not just a chat reply. If you're in "
    "immediate danger, please contact your local emergency number. "
    "You can also reach a crisis line to talk to someone right now -- "
    "in Pakistan, Umang has a helpline at 0311 7786264. If you're "
    "elsewhere, please look up a local crisis line or go to your "
    "nearest emergency room. You don't have to go through this alone."
)

FALLBACK_REPLY = (
    "I'm here, but I'm currently running without a connected AI model "
    "(no ANTHROPIC_API_KEY is configured on the server), so I can't "
    "have a full conversation yet. Your message has still been saved. "
    "In the meantime, you can use the PCOS Detection and Protection "
    "tabs directly, or ask your project supervisor to help set up the "
    "AI key so this check-in can respond properly."
)

SYSTEM_PROMPT = """You are the AI check-in assistant inside Wellness Saheli, \
a women's reproductive health app. Your role is to have a warm, brief, \
supportive check-in conversation about cycle symptoms, lifestyle \
(exercise, diet, sleep), and general wellbeing.

Strict rules:
- Never diagnose any physical or mental health condition (including \
PCOS, anxiety, or depression). You may reflect what the user shares \
and suggest they use the app's PCOS Detection tool or talk to a \
doctor, but never state or imply a diagnosis yourself.
- Keep replies short (2-4 sentences) and conversational, not clinical.
- If the user describes something urgent or medically concerning \
(e.g. severe pain, heavy bleeding, thoughts of self-harm), gently \
encourage them to seek professional care rather than trying to \
resolve it yourself.
- Do not give specific drug dosages or medical treatment instructions.
"""


def _contains_crisis_language(message: str) -> bool:
    lowered = message.lower()
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


def _call_claude(message: str, history: list[dict]) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = []
    for turn in history[-10:]:  # keep the request small; last 10 turns
        role = "user" if turn.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": turn.get("message", "")})
    messages.append({"role": "user", "content": message})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def get_reply(message: str, history: list[dict]) -> dict:
    """Returns the dict shape ChatTurnResult.fromJson expects:
    reply, profile_updates, suggested_tab, suggested_tab_reason,
    crisis_concern."""

    if _contains_crisis_language(message):
        return {
            "reply": CRISIS_REPLY,
            "profile_updates": None,
            "suggested_tab": None,
            "suggested_tab_reason": None,
            "crisis_concern": True,
        }

    if "ANTHROPIC_API_KEY" not in os.environ:
        return {
            "reply": FALLBACK_REPLY,
            "profile_updates": None,
            "suggested_tab": None,
            "suggested_tab_reason": None,
            "crisis_concern": False,
        }

    try:
        reply_text = _call_claude(message, history)
    except Exception:
        # Network/API failure -- fail safely rather than propagating a
        # 500 that would show the user a raw error in the chat screen.
        return {
            "reply": (
                "Sorry, I couldn't reach the AI service just now. "
                "Please try again in a moment."
            ),
            "profile_updates": None,
            "suggested_tab": None,
            "suggested_tab_reason": None,
            "crisis_concern": False,
        }

    suggested_tab = None
    suggested_tab_reason = None
    lowered = message.lower()
    if any(w in lowered for w in ["irregular period", "missed period", "acne", "hair growth", "weight gain"]):
        suggested_tab = "pcos"
        suggested_tab_reason = "These symptoms can be worth checking with the PCOS Detection tool."
    elif any(w in lowered for w in ["contracept", "birth control", "pregnan", "iud", "condom"]):
        suggested_tab = "protection"
        suggested_tab_reason = "The Protection tab can help you compare contraceptive options."

    return {
        "reply": reply_text,
        "profile_updates": None,
        "suggested_tab": suggested_tab,
        "suggested_tab_reason": suggested_tab_reason,
        "crisis_concern": False,
    }