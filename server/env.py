import os
from typing import Dict, Optional
from openenv.core.env_server import Action, Observation, State, Environment

class TicketAction(Action):
    category: str = "other"  # 'billing', 'technical', 'account', 'other'
    priority: str = "normal" # 'low', 'normal', 'high', 'urgent'
    action: str = "ignore"   # 'respond', 'escalate', 'ignore'

class TicketObservation(Observation):
    next_ticket: Optional[Dict[str, str]] = None
    remaining_tickets_count: int = 0
    message: str = ""
    # Inherits: done: bool, reward: float, error: Optional[str]

class TicketState(State):
    processed_count: int = 0
    total_count: int = 0
    current_task: str = "easy"

# Tickets for tasks
EASY_TICKETS = [
    {"id": "T1", "subject": "Forgot password", "body": "I can't log in.", "expected_category": "account"},
    {"id": "T2", "subject": "Invoice for April", "body": "Need billing statement.", "expected_category": "billing"},
    {"id": "T3", "subject": "App crash", "body": "The app crashes on startup.", "expected_category": "technical"}
]

MEDIUM_TICKETS = [
    {"id": "T4", "subject": "URGENT Data missing", "body": "Production data is gone!", "expected_category": "technical", "expected_priority": "urgent"},
    {"id": "T5", "subject": "Change email", "body": "Update my email.", "expected_category": "account", "expected_priority": "low"},
    {"id": "T6", "subject": "Refund now", "body": "Overcharged. Refund ASAP.", "expected_category": "billing", "expected_priority": "high"},
    {"id": "T7", "subject": "Typo", "body": "Typo on front page.", "expected_category": "other", "expected_priority": "low"},
    {"id": "T8", "subject": "500 error", "body": "API gives 500 error.", "expected_category": "technical", "expected_priority": "high"}
]

HARD_TICKETS = [
    {"id": "T9", "subject": "OOM Error", "body": "Server out of memory.", "expected_category": "technical", "expected_priority": "urgent", "expected_action": "escalate"},
    {"id": "T10", "subject": "Account locked", "body": "Locked out again.", "expected_category": "account", "expected_priority": "high", "expected_action": "respond"},
    {"id": "T11", "subject": "Cancel pro", "body": "Downgrade me.", "expected_category": "billing", "expected_priority": "normal", "expected_action": "respond"},
    {"id": "T12", "subject": "1000 seats", "body": "Want to buy enterprise.", "expected_category": "billing", "expected_priority": "high", "expected_action": "escalate"},
    {"id": "T13", "subject": "XSS Bug", "body": "Found a security bug.", "expected_category": "technical", "expected_priority": "urgent", "expected_action": "escalate"},
    {"id": "T14", "subject": "Hey", "body": "Cool app.", "expected_category": "other", "expected_priority": "low", "expected_action": "ignore"}
]

class SupportTriageEnv(Environment):
    def __init__(self):
        super().__init__()
        self.task_name = "easy"
        self.tickets = []
        self.current_index = 0
        self.total_tickets = 0

    def reset(self) -> TicketObservation:
        self.task_name = os.getenv("OPENENV_TASK", "easy")
        
        if self.task_name == "easy":
            self.tickets = list(EASY_TICKETS)
        elif self.task_name == "medium":
            self.tickets = list(MEDIUM_TICKETS)
        elif self.task_name == "hard":
            self.tickets = list(HARD_TICKETS)
        else:
            self.tickets = list(EASY_TICKETS)
            
        self.current_index = 0
        self.total_tickets = len(self.tickets)

        next_ticket = None
        if self.total_tickets > 0:
            next_ticket = {
                "id": self.tickets[0]["id"],
                "subject": self.tickets[0]["subject"],
                "body": self.tickets[0]["body"]
            }

        return TicketObservation(
            next_ticket=next_ticket,
            remaining_tickets_count=self.total_tickets,
            message=f"Environment reset. Task: {self.task_name}",
            done=False,
            reward=0.0
        )

    def step(self, action: TicketAction) -> TicketObservation:
        if self.current_index >= self.total_tickets:
            return TicketObservation(
                next_ticket=None,
                remaining_tickets_count=0,
                message="Episode finished.",
                done=True,
                reward=0.0
            )

        current_ticket = self.tickets[self.current_index]
        reward = 0.0
        
        cat = action.category.lower().strip()
        pri = action.priority.lower().strip()
        act = action.action.lower().strip()
        
        step_base_reward = 1.0 / self.total_tickets

        if self.task_name == "easy":
            if cat == current_ticket.get("expected_category", ""):
                reward = step_base_reward
                
        elif self.task_name == "medium":
            if cat == current_ticket.get("expected_category", ""):
                reward += step_base_reward * 0.5
            if pri == current_ticket.get("expected_priority", "normal"):
                reward += step_base_reward * 0.5
                
        elif self.task_name == "hard":
            if cat == current_ticket.get("expected_category", ""):
                reward += step_base_reward * 0.334
            if pri == current_ticket.get("expected_priority", "normal"):
                reward += step_base_reward * 0.333
            if act == current_ticket.get("expected_action", "ignore"):
                reward += step_base_reward * 0.333

        self.current_index += 1
        done = self.current_index >= self.total_tickets

        next_ticket = None
        if not done:
            t = self.tickets[self.current_index]
            next_ticket = {"id": t["id"], "subject": t["subject"], "body": t["body"]}

        return TicketObservation(
            next_ticket=next_ticket,
            remaining_tickets_count=self.total_tickets - self.current_index,
            message="Action processed.",
            done=done,
            reward=round(reward, 4)
        )

    @property
    def state(self) -> TicketState:
        return TicketState(
            processed_count=self.current_index,
            total_count=self.total_tickets,
            current_task=self.task_name
        )
