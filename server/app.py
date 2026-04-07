from openenv.core.env_server import create_fastapi_app
from server.env import SupportTriageEnv, TicketAction, TicketObservation
import uvicorn

app = create_fastapi_app(SupportTriageEnv, TicketAction, TicketObservation)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
