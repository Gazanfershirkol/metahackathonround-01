import os
import json
import textwrap
from openai import OpenAI
from server.env import SupportTriageEnv, TicketAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

SYSTEM_PROMPT = """
You are an autonomous AI Support Agent.
Your task is to triage customer support tickets by providing exactly ONE JSON object as your action.
The action JSON must have these keys:
- "category": string, one of ['billing', 'technical', 'account', 'other']
- "priority": string, one of ['low', 'normal', 'high', 'urgent']
- "action": string, one of ['respond', 'escalate', 'ignore']

Output ONLY valid JSON, nothing else. Example:
{
  "category": "technical",
  "priority": "high",
  "action": "escalate"
}
"""

def parse_model_action(response_text: str) -> dict:
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response_text[start:end]
            return json.loads(json_str)
    except Exception:
        pass
    return {"category": "other", "priority": "normal", "action": "ignore"}

def main():
    print(f"Using Model: {MODEL_NAME}")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Key available: {bool(API_KEY)}")
    if not API_KEY:
        print("WARNING: API_KEY/HF_TOKEN not set, but inference script depends on it.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    # Evaluate on all 3 tasks
    tasks = ["easy", "medium", "hard"]
    
    for task_name in tasks:
        print(f"\n{'='*40}")
        print(f"Starting Task: {task_name.upper()}")
        print(f"{'='*40}")
        
        # Instantiate environment directly
        env = SupportTriageEnv()
        os.environ["OPENENV_TASK"] = task_name
        
        obs_state = env.reset()
        done = False
        step = 1
        total_reward = 0.0
        
        while not done:
            if obs_state.next_ticket is None:
                break
                
            ticket = obs_state.next_ticket
            user_prompt = textwrap.dedent(f"""
            Current Ticket ({task_name.upper()} Difficulty):
            Subject: {ticket['subject']}
            Body: {ticket['body']}
            Remaining Tickets: {obs_state.remaining_tickets_count}
            
            Determine the correct category, priority, and action. Follow system instructions strictly.
            """)
            
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=150,
                )
                response_text = completion.choices[0].message.content or "{}"
            except Exception as e:
                print(f"Model request failed: {e}")
                response_text = "{}"
                
            action_dict = parse_model_action(response_text)
            
            print(f"Step {step} | Ticket: '{ticket['subject']}'")
            print(f"Model Action: {action_dict}")
            
            action_obj = TicketAction(
                category=action_dict.get("category", "other"),
                priority=action_dict.get("priority", "normal"),
                action=action_dict.get("action", "ignore")
            )
            
            obs_state = env.step(action_obj)
            total_reward += obs_state.reward
            print(f"Reward: {obs_state.reward:.4f} | Done: {obs_state.done}")
            print("-" * 30)
            
            done = obs_state.done
            step += 1
            
        print(f"\nFinal Total Reward for {task_name.upper()}: {total_reward:.4f}")

if __name__ == "__main__":
    main()
