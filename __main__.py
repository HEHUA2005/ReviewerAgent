"""
Main entry point for the Reviewer Agent.

This module sets up the A2A server and defines the agent card and skills.
"""
import logging
import os
import sys
import uvicorn
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_executor import ReviewerAgentExecutor
from config import (
    HOST, PORT, AGENT_URL, 
    get_agent_skills,
    get_public_agent_card,
    validate_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reviewer_agent.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    logger.info("Starting Reviewer Agent")
    load_dotenv() 
    # Validate configuration
    validate_config()
    
    task_store = InMemoryTaskStore()
    
    agent_executor = ReviewerAgentExecutor()
    
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
    )
    
    agent_card = get_public_agent_card()
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # Print startup message
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                     REVIEWER AGENT                           ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Server URL: {AGENT_URL}
║  Version: {agent_card.version}
║                                                              ║
║  Skills:                                                     ║
""")
    
    for skill in agent_card.skills:
        desc = skill.description[:40] + "..." if len(skill.description) > 40 else skill.description
        padding = ' ' * max(0, (10 - len(skill.name)))
        print(f"║    - {skill.name}: {desc}{padding}║")
    
    print(f"""║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Start server
    uvicorn.run(server.build(), host=HOST, port=PORT)

if __name__ == "__main__":
    main()