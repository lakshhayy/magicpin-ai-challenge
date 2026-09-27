import os
import time
import json
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel
import httpx

from conversation_handlers import handle_reply

app = FastAPI(title="Magicpin AI Challenge - Vera Bot")
START_TIME = time.time()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT & STATE
# =============================================================================
GROQ_API_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
GROQ_MODEL = "openai/gpt-oss-120b"
_key_index = 0  # Round-robin counter

contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}
conversations: Dict[str, List[Dict[str, Any]]] = {}

# =============================================================================
# ROUTING & PROMPTS
# =============================================================================
SYSTEM_PROMPT_MERCHANT = """You are 'Vera', an expert AI assistant that engages merchants on WhatsApp to help them grow their business.

BEFORE COMPOSING, follow these steps:
STEP 1 — EXTRACT from the JSON context:
  - owner_first_name, business_name, city, languages[]
  - All numbers from performance{} (views, calls, ctr, delta_7d percentages)
  - All numbers from the trigger payload (delta_pct, vs_baseline, days_remaining, renewal_amount, dates)
  - Active offers[] with exact titles and prices
  - conversation_history[] to understand the ongoing thread
  - peer_stats from the category context (avg_rating, avg_review_count, avg_views_30d)

STEP 2 — COMPOSE using ONLY the extracted data. Rules:
1. SPECIFICITY (MOST IMPORTANT — 10/10 target):
   - You must include at least three specific data points in the message (exact view counts, exact percentages, specific INR prices, or exact dates). Never use vague terms like 'increased' or 'many'.
2. CATEGORY FIT: Adapt your tone based on the category: use a clinical, peer-to-peer tone with 'Dr.' for dentists; a warm, practical tone for salons; and an operator-to-operator tone for restaurants.
3. MERCHANT FIT: Address the merchant using their exact owner name. Include a localized greeting using their specified preferred languages. Reference their specific active offers rather than generic suggestions.
4. TRIGGER RELEVANCE: Explicitly state the trigger event in the first sentence to establish urgency (e.g., 'Because your Pro plan expires in 12 days...', 'Since views dropped 30% today...'). Tie the proposed solution directly to this exact event.
5. ENGAGEMENT COMPULSION: End the message with a single, low-friction Yes/No question. Use loss aversion to drive the CTA (e.g., 'Reply YES within 24 hours to launch this campaign before you lose weekend footfall to competitors').
6. NO FABRICATION: If a number is NOT in the JSON, do NOT use it. Zero tolerance.
7. NO URLS: Do not include http/https links.

PERFECT EXAMPLE (DO NOT COPY DATA, COPY STRUCTURE):
Context JSON: {"business_name": "Dr. Raj Clinic", "performance": {"views_30d": 1200, "delta_views_pct": -0.15}}
Good Body: "Hi Dr. Raj! Your clinic logged 1,200 views this month, down 15% week-on-week."
Bad Body: "Your views dropped a lot recently, let's fix it!" (Lacks numbers)
Bad Body: "Your clinic got 1,200 views, and your calls dropped 10%!" (Fabricated 10% calls)

OUTPUT FORMAT:
Respond with a pure JSON object containing:
- body: The WhatsApp message text
- cta: Call to action type (e.g., "binary_yes_no", "binary_confirm_cancel", "open_ended", "none")
- template_name: A generic template name (e.g. "vera_merchant_v1")
- template_params: A list of string parameters used in the template
- send_as: "vera"
- rationale: A short explanation of why this message fits the 5-dimension scoring rubric."""

SYSTEM_PROMPT_CUSTOMER = """You are composing a WhatsApp message to a customer ON BEHALF OF a merchant.

BEFORE COMPOSING, follow these steps:
STEP 1 — EXTRACT from the JSON context:
  - customer name, language preference
  - merchant business_name, owner_first_name
  - All dates, prices, slot times from the trigger payload
  - Active offers[] with exact titles and prices from merchant context

STEP 2 — COMPOSE using ONLY the extracted data. Rules:
1. VOICE: Speak AS the merchant (e.g., "Hi Priya, Dr. Meera's Dental Clinic here"). Never mention Vera.
2. SPECIFICITY (MOST IMPORTANT — 10/10 target):
   - Quote ONLY numbers from the JSON. Do NOT calculate or infer.
   - Include exact slot times, exact prices, exact dates from the trigger payload.
   - Reference the specific service_due, last_service_date, due_date if present.
3. CTA: End with a numbered choice (e.g., "Reply 1 for Wed 5 Nov 6pm, 2 for Thu 6 Nov 5pm").
4. NO FABRICATION: If a number is NOT in the JSON, do NOT use it. Zero tolerance.
5. NO URLS: Do not include http/https links.
6. LANGUAGE: Use the customer's language preference exactly (e.g., if "hi-en", write in Hinglish).

PERFECT EXAMPLE (DO NOT COPY DATA, COPY STRUCTURE):
Context JSON: {"customer_name": "Rohan", "trigger": {"payload": {"due_date": "2026-11-12", "available_slots": [{"date":"Wed 5 Nov", "time":"6pm"}]}}}
Good Body: "Hi Rohan, Dr. Raj Clinic here! Your next visit is due on 2026-11-12. Reply 1 for Wed 5 Nov 6pm."
Bad Body: "Hi Rohan, it's time for your visit! We have discounts!" (Generic)
Bad Body: "Your next visit is due soon. Reply 1 for tomorrow morning." (Fabricated slot)

OUTPUT FORMAT:
Respond with a pure JSON object containing:
- body: The WhatsApp message text
- cta: Call to action type (e.g., "multi_choice_slot", "binary_yes_no", "none")
- template_name: A generic template name (e.g. "merchant_customer_v1")
- template_params: A list of string parameters used in the template
- send_as: "merchant_on_behalf"
- rationale: A short explanation of why this message fits the 5-dimension scoring rubric."""

async def call_llm(prompt: str, system_prompt: str) -> dict:
    global _key_index
    if not GROQ_API_KEYS:
        logger.warning("GROQ_API_KEYS not set!")
        return {}
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    for attempt in range(6):
        # Round-robin key selection
        key = GROQ_API_KEYS[_key_index % len(GROQ_API_KEYS)]
        _key_index += 1
        
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                # Remove markdown if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                    
                content = content.strip()
                
                logger.info(f"LLM raw content length: {len(content)}")
                return json.loads(content)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < 5:
                logger.warning(f"429 on key ...{key[-4:]}, rotating to next key (attempt {attempt+1}/6)")
                await asyncio.sleep(5.0)  # Short sleep then try next key
            else:
                logger.error(f"LLM HTTP Error: {e}")
                return {}
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {}

# =============================================================================
# API MODELS
# =============================================================================
class ContextPushBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TIME), "contexts_loaded": counts}

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Team Lakshay",
        "team_members": ["Lakshay Kaushik"],
        "model": GROQ_MODEL,
        "approach": "Advanced routing layer with post-LLM validation and strict state machine.",
        "contact_email": "hello@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/v1/context")
async def push_context(body: ContextPushBody):
    key = (body.scope, body.context_id)
    current = contexts.get(key)
    
    if current and current["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}
        
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": datetime.utcnow().isoformat() + "Z"}

@app.post("/v1/tick")
async def tick(body: TickBody):
    async def process_trigger(trg_id: str):
        trg = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trg:
            logger.warning(f"Trigger {trg_id} not found in contexts!")
            return None
        
        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")
        
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload") if merchant_id else None
        if not merchant:
            logger.warning(f"Merchant {merchant_id} not found for trigger {trg_id}")
            return None
            
        customer = contexts.get(("customer", customer_id), {}).get("payload") if customer_id else None
        
        category_slug = merchant.get("category_slug")
        category = contexts.get(("category", category_slug), {}).get("payload") if category_slug else None
        if not category:
            logger.warning(f"Category {category_slug} not found for merchant {merchant_id}")
            return None
            
        logger.info(f"Routing trigger {trg_id} for merchant {merchant_id} (Scope: {trg.get('scope')})")
        
        # ROUTING LAYER
        if trg.get("scope") == "customer":
            sys_prompt = SYSTEM_PROMPT_CUSTOMER
        else:
            sys_prompt = SYSTEM_PROMPT_MERCHANT
            
        prompt = json.dumps({
            "category": category,
            "merchant": merchant,
            "customer": customer,
            "trigger": trg
        }, indent=2)
        
        llm_resp = await call_llm(prompt, sys_prompt)
        
        if not llm_resp:
            return None
            
        body_text = llm_resp.get("body", "")
        
        # POST-LLM VALIDATION: Remove URLs if LLM hallucinated them
        if "http://" in body_text or "https://" in body_text:
            logger.warning("LLM hallucinated a URL. Removing it to avoid penalty.")
            body_text = body_text.replace("http://", "[LINK_REMOVED]").replace("https://", "[LINK_REMOVED]")
            
        logger.info(f"\n{'='*60}\n📩 MESSAGE for {trg_id}:\n{body_text}\n{'='*60}")
        return {
            "conversation_id": f"conv_{merchant_id}_{trg_id}",
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": llm_resp.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": llm_resp.get("template_name", "generic_v1"),
            "template_params": llm_resp.get("template_params", []),
            "body": body_text,
            "cta": llm_resp.get("cta", "open_ended"),
            "suppression_key": trg.get("suppression_key", f"fallback_{trg_id}"),
            "rationale": llm_resp.get("rationale", "Generated by LLM via Routing Layer")
        }

    actions = []
    for tid in body.available_triggers:
        result = await process_trigger(tid)
        if result:
            actions.append(result)
        
        # Sleep 3s between triggers (4 keys = ~12 RPM effective, safely under limits)
        await asyncio.sleep(3.0)
            
    return {"actions": actions}

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_history = conversations.setdefault(body.conversation_id, [])
    conv_history.append({"from": body.from_role, "msg": body.message})
    
    merchant = contexts.get(("merchant", body.merchant_id), {}).get("payload") if body.merchant_id else None
    
    logger.info(f"Processing reply for conversation {body.conversation_id}, turn {body.turn_number}")
    response = await handle_reply(merchant, conv_history, call_llm)
    
    # Keep track of Vera's outgoing messages in the history
    if response.get("action") == "send":
        conv_history.append({"from": "vera", "msg": response.get("body")})
        
    return response
