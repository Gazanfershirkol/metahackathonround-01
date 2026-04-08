"""
Inference Script — Support Ticket Triage
=========================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import json
import time
import textwrap
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — reads from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")

# Environment base URL (the deployed HF Space or local Docker)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS = 20          # safety cap per task
TEMPERATURE = 0.0
MAX_TOKENS = 200

SYSTEM_PROMPT = textwrap.dedent("""\
You are an autonomous AI Support Agent performing ticket triage.
For each support ticket, you must decide:
1. Category: one of 'billing', 'technical', 'account', 'other'
2. Priority: one of 'low', 'normal', 'high', 'urgent'
3. Action: one of 'respond', 'escalate', 'ignore'

Respond with ONLY a valid JSON object. No explanation, no markdown. Example:
{"category": "technical", "priority": "high", "action": "escalate"}
""")


# ---------------------------------------------------------------------------
# Helper — parse model output to action dict
# ---------------------------------------------------------------------------
def parse_model_action(response_text: str) -> dict:
    """Extract the first JSON object from model response."""
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response_text[start:end])
    except Exception:
        pass
    return {"category": "other", "priority": "normal", "action": "ignore"}


# ---------------------------------------------------------------------------
# HTTP helpers — talk to the environment server
# ---------------------------------------------------------------------------
def env_reset(task_name: str) -> dict:
    """POST /reset to the environment server."""
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    """POST /step to the environment server."""
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Direct-mode helpers  (fallback when no server is running)
# ---------------------------------------------------------------------------
def run_direct(task_name: str, client: OpenAI):
    """Run the environment in-process (no HTTP server needed)."""
    from server.env import SupportTriageEnv, TicketAction

    env = SupportTriageEnv()
    os.environ["OPENENV_TASK"] = task_name

    obs = env.reset()
    done = False
    step_num = 0
    total_reward = 0.0
    start_time = time.time()

    # ── [START] ──────────────────────────────────────────────────
    print(f"[START] task={task_name} model={MODEL_NAME}")

    while not done and step_num < MAX_STEPS:
        if obs.next_ticket is None:
            break

        ticket = obs.next_ticket
        step_num += 1

        user_prompt = textwrap.dedent(f"""\
        Support Ticket to Triage:
        Subject: {ticket['subject']}
        Body: {ticket['body']}
        Remaining tickets in queue: {obs.remaining_tickets_count}

        Classify this ticket. Respond with JSON only.""")

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or "{}"
        except Exception as exc:
            print(f"[STEP] step={step_num} error=\"Model request failed: {exc}\"")
            response_text = "{}"

        action_dict = parse_model_action(response_text)

        action_obj = TicketAction(
            category=action_dict.get("category", "other"),
            priority=action_dict.get("priority", "normal"),
            action=action_dict.get("action", "ignore"),
        )

        obs = env.step(action_obj)
        total_reward += obs.reward
        done = obs.done

        # ── [STEP] ──────────────────────────────────────────────
        print(
            f"[STEP] step={step_num} "
            f"ticket_id={ticket.get('id', 'N/A')} "
            f"action={json.dumps(action_dict)} "
            f"reward={obs.reward:.4f} "
            f"done={done}"
        )

    elapsed = round(time.time() - start_time, 2)

    # ── [END] ────────────────────────────────────────────────────
    print(
        f"[END] task={task_name} "
        f"total_reward={total_reward:.4f} "
        f"steps={step_num} "
        f"elapsed={elapsed}s"
    )
    return total_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  Support Ticket Triage - Baseline Inference")
    print("=" * 60)
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  API_KEY set  : {bool(API_KEY)}")
    print("=" * 60)

    if not API_KEY:
        print("WARNING: HF_TOKEN / API_KEY not set. LLM calls will fail.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    tasks = ["easy", "medium", "hard"]
    results = {}

    for task_name in tasks:
        score = run_direct(task_name, client)
        results[task_name] = score

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BASELINE RESULTS")
    print("=" * 60)
    for task, score in results.items():
        print(f"  {task:8s} -> {score:.4f}")
    avg = sum(results.values()) / len(results)
    print(f"  {'average':8s} -> {avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
