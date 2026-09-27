import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_REPLY = """You are 'Vera', evaluating a multi-turn conversation with a merchant.
You must decide the next action based on their latest reply.

CRITICAL RULES:
1. INTENT HANDOFF: If the merchant agreed to your previous pitch (e.g., "Yes", "Let's do it", "Okay"), STOP pitching. Switch immediately to Action Mode (e.g., "Great, I'll draft the post now. Reply CONFIRM to publish").
2. QUESTIONS: If they asked a clarifying question, answer it concisely using data from the merchant context if possible, and reiterate a simple CTA.
3. REFUSAL: If they politely decline but are open to future messages, return action "wait" and set a wait_seconds of 86400 (1 day).
4. AUTO-REPLIES: If the message looks like a generic auto-responder, return action "wait" with 7200 seconds.

OUTPUT FORMAT:
Respond with a pure JSON object containing:
- action: "send", "wait", or "end"
- body: The WhatsApp message text (only if action="send")
- cta: Call to action type (only if action="send")
- wait_seconds: Integer seconds (only if action="wait")
- rationale: A short explanation for your decision."""

async def handle_reply(merchant: dict, conv_history: list, call_llm_func) -> dict:
    latest_msg = conv_history[-1].get("msg", "").lower()
    
    # 1. Hostility / Opt-out Interceptor
    hostile_keywords = ["stop", "unsubscribe", "spam", "don't message", "do not message"]
    if any(k in latest_msg for k in hostile_keywords):
        logger.info("Hostile interceptor triggered.")
        return {
            "action": "end",
            "rationale": "Explicit opt-out or hostility detected."
        }
        
    # 2. Auto-Reply Interceptor (Heuristic fallback)
    auto_keywords = ["thank you for contacting", "away message", "automated message", "we will get back", "out of office"]
    if any(k in latest_msg for k in auto_keywords):
        logger.info("Auto-reply interceptor triggered.")
        return {
            "action": "wait",
            "wait_seconds": 7200,
            "rationale": "Automated business reply detected by heuristics. Backing off."
        }
        
    # 3. Dynamic LLM Evaluation for Intent Handoff & Replies
    prompt = json.dumps({
        "merchant": merchant,
        "conversation_history": conv_history
    }, indent=2)
    
    llm_resp = await call_llm_func(prompt, SYSTEM_PROMPT_REPLY)
    
    if not llm_resp:
        return {"action": "wait", "wait_seconds": 3600, "rationale": "LLM failure. Safe wait."}
        
    action = llm_resp.get("action", "wait")
    
    if action == "send":
        return {
            "action": "send",
            "body": llm_resp.get("body", ""),
            "cta": llm_resp.get("cta", "open_ended"),
            "rationale": llm_resp.get("rationale", "")
        }
    elif action == "wait":
        return {
            "action": "wait",
            "wait_seconds": llm_resp.get("wait_seconds", 3600),
            "rationale": llm_resp.get("rationale", "")
        }
    else:
        return {
            "action": "end",
            "rationale": llm_resp.get("rationale", "")
        }
