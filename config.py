import os
from typing import List
from a2a.types import AgentCapabilities, AgentSkill, AgentCard
from dotenv import load_dotenv

load_dotenv()

# Server Configuration
HOST = os.getenv("AGENT_HOST", "0.0.0.0")
PORT = int(os.getenv("AGENT_PORT", "9997"))
AGENT_URL = os.getenv("AGENT_URL", f"http://localhost:{PORT}/")

# LLM Configuration
API_KEY = os.getenv("API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "16000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.01"))

# OpenAI API Configuration
BASE_URL = os.getenv("BASE_URL")  # Custom OpenAI API endpoint URL

# Paper Search Configuration
ARXIV_CATEGORIES = os.getenv("ARXIV_CATEGORIES", "cs.AI,cs.LG,cs.CL")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

# Review Configuration
REVIEW_CRITERIA = os.getenv("REVIEW_CRITERIA", "academic_peer_review")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# Agent Skills Configuration
def get_agent_skills() -> List[AgentSkill]:
    """Define all agent skills."""
    return [
        AgentSkill(
            id="review_pdf_paper",
            name="Review PDF Paper",
            description="Review an academic paper provided as a PDF file. Performs comprehensive analysis with structured scoring for methodology, novelty, clarity, and significance.",
            tags=["review", "academic", "pdf", "analysis"],
            examples=[
                "Review this paper on transformer architectures",
                "Analyze this PDF paper",
                "Evaluate this research paper",
            ],
        ),
        AgentSkill(
            id="search_and_review",
            name="Search and Review Paper",
            description="Search for academic papers based on a natural language description, present options for selection, and review the selected paper.",
            tags=["search", "review", "academic", "analysis"],
            examples=[
                "Find papers about transformer attention mechanisms",
                "Search for recent papers on reinforcement learning",
                "Look for research on graph neural networks and review the best one",
            ],
        ),
        AgentSkill(
            id="get_review_template",
            name="Get Review Template",
            description="Return the review criteria template with scoring guidelines and explanation of the review process.",
            tags=["template", "criteria", "guidelines"],
            examples=[
                "Show me the review criteria",
                "What scoring system do you use?",
                "How do you evaluate papers?",
            ],
        ),
    ]



def get_public_agent_card() -> AgentCard:
    """Create the public agent card."""
    return AgentCard(
        name="Reviewer Agent",
        description="A specialized AI agent focused on reviewing academic papers. Can accept papers directly as PDF files or search for papers based on natural language descriptions.",
        url=AGENT_URL,
        version="1.0.0",
        defaultInputModes=["text", "file"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=get_agent_skills(),
        supportsAuthenticatedExtendedCard=False,
    )


# Validate configuration
def validate_config():
    """Validate the configuration and print warnings for missing values."""
    if not API_KEY:
        print(
            "WARNING: API_KEY environment variable is not set. LLM functionality will be limited."
        )


    if not SEMANTIC_SCHOLAR_API_KEY:
        print(
            "WARNING: SEMANTIC_SCHOLAR_API_KEY environment variable is not set. Semantic Scholar search will be limited."
        )
