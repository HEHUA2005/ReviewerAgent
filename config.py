import os
from typing import List
from a2a.types import AgentCapabilities, AgentSkill, AgentCard
from dotenv import load_dotenv

load_dotenv()

# Server Configuration
HOST = os.getenv("AGENT_HOST", "0.0.0.0")
PORT = int(os.getenv("AGENT_PORT", "9998"))
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
            description="Review an academic paper provided as a PDF file. Performs comprehensive analysis with structured scoring (1-10) for methodology, novelty, clarity, and significance. Automatically extracts text, metadata, and generates detailed review report.",
            tags=["review", "academic", "pdf", "analysis", "scoring"],
            examples=[
                "Review this paper on transformer architectures",
                "Analyze this PDF paper",
                "Evaluate this research paper",
                "[Upload PDF file for review]",
            ],
        ),
        AgentSkill(
            id="search_and_review",
            name="Search and Review Paper",
            description="Search for academic papers from arXiv and Semantic Scholar based on natural language queries. Returns numbered list of results (up to 5 papers) with titles, authors, abstracts. Supports multi-step workflow: search → select → download → review. Papers are cached locally for efficiency.",
            tags=[
                "search",
                "review",
                "academic",
                "arxiv",
                "semantic-scholar",
                "multi-step",
            ],
            examples=[
                "Find papers about transformer attention mechanisms",
                "Search for recent papers on reinforcement learning",
                "找一些关于GAN的论文",
                "Select paper 2",
                "选第一篇",
            ],
        ),
        AgentSkill(
            id="review_paper_url",
            name="Review Paper from URL",
            description="Review a paper directly from its URL (supports arXiv and other academic paper URLs). Automatically downloads the PDF, caches it locally, and generates comprehensive review with scoring and detailed analysis.",
            tags=["review", "url", "arxiv", "download", "analysis"],
            examples=[
                "https://arxiv.org/abs/1706.03762",
                "Review this paper: https://arxiv.org/pdf/2005.14165.pdf",
                "Analyze the paper at https://arxiv.org/abs/2010.11929",
            ],
        ),
        AgentSkill(
            id="get_review_template",
            name="Get Review Template",
            description="Display the review criteria template with detailed scoring guidelines (1-10 scale) for methodology, novelty, clarity, and significance. Explains the review process and evaluation standards used by the agent.",
            tags=["template", "criteria", "guidelines", "scoring", "standards"],
            examples=[
                "Show me the review criteria",
                "What scoring system do you use?",
                "How do you evaluate papers?",
                "评分标准是什么？",
            ],
        ),
        AgentSkill(
            id="greeting_and_help",
            name="Greeting and Help",
            description="Handle greetings and provide information about the agent's capabilities. Explains available functions and how to use them effectively.",
            tags=["greeting", "help", "introduction", "capabilities"],
            examples=[
                "Hello",
                "你好",
                "What can you do?",
                "How does this work?",
            ],
        ),
    ]


def get_public_agent_card() -> AgentCard:
    """Create the public agent card."""
    return AgentCard(
        name="Reviewer Agent",
        description="""A specialized AI agent for academic paper review and search using the A2A protocol.

A2A MESSAGE FORMAT:
The agent accepts messages following the A2A (Agent-to-Agent) protocol structure:
{
  "message": {
    "role": "user",
    "parts": [
      {"kind": "text", "text": "your query here"} // for text input
      // OR
      {"kind": "file", "file": {"name": "paper.pdf", "bytes": "base64_encoded_data"}} // for PDF input
    ]
  }
}

INPUT TYPES:
1. TEXT MESSAGES (kind: "text"):
   - Paper search queries: "Find papers about transformer attention mechanisms"
   - Paper selection: "Select paper 2" or "Choose the first one"
   - Direct URL review: "https://arxiv.org/abs/1706.03762"
   - Review template request: "Show me the review criteria"

2. FILE MESSAGES (kind: "file"):
   - PDF files with academic papers
   - File must include: name (ending in .pdf) and base64-encoded bytes
   - Automatic text extraction and comprehensive analysis

OUTPUT FORMAT:
All responses are sent as text messages through the A2A event stream:
- Paper search results: Numbered list with title, authors, year, abstract, and selection prompt
- Review reports: Structured markdown with:
  * Paper metadata (title, authors, year, source)
  * Scores (1-10): Methodology, Novelty, Clarity, Significance
  * Detailed analysis: Summary, Strengths, Weaknesses, Questions
- Review template: Scoring criteria and guidelines
- Status updates and error messages

WORKFLOW CONTEXT:
The agent maintains context across messages for multi-step workflows:
1. Search → Returns numbered results → Awaits selection
2. Selection → Downloads/processes paper → Returns comprehensive review
3. Direct PDF/URL → Immediate review generation

TECHNICAL DETAILS:
- Protocol: A2A JSON-RPC 2.0
- Streaming: Supports real-time response streaming
- Context: Maintains task state using context_id for conversation continuity""",
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
