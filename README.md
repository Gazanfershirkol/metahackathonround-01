# Support Ticket Triage Environment

## Overview
This OpenEnv environment simulates a real-world Customer Support system where an agent acts as a Level 1 Support representative. The primary objective is to manage a queue of incoming customer request tickets by correctly categorizing each issue, assigning an appropriate priority, and optionally selecting a preliminary action.

## Action Space
`TicketAction` (Pydantic model):
- `category` (str): 'billing', 'technical', 'account', 'other'
- `priority` (str): 'low', 'normal', 'high', 'urgent'
- `action` (str): 'respond', 'escalate', 'ignore'

## Observation Space
`TicketObservation` (Pydantic model):
- `next_ticket` (dict/str): Information about the current ticket being evaluated, including 'id', 'subject', and 'body'.
- `remaining_tickets_count` (int): Number of tickets left in the queue.

## Tasks
1. **Easy**: Process 3 clear-cut tickets. Evaluating categorization only (priority/action logic defaults if simple).
2. **Medium**: Process 5 tickets requiring both categorization and logical prioritization (e.g. angry customer = high priority).
3. **Hard**: Process 6 complex tickets needing categorization, prioritization, and the correct follow-up action.

## Quick Start
1. Build the container: `docker build -t openenv-support-triage .`
2. Run the environment: `docker run -p 7860:7860 openenv-support-triage`
3. Run baseline agent: `python inference.py` (ensure `API_BASE_URL`, `HF_TOKEN`, `MODEL_NAME` are set).
