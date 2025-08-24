"""
Agent Executor module for the Reviewer Agent.
"""

import asyncio
import base64
import json
import logging
import os
import re
import uuid
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.types import (
    TaskStatusUpdateEvent,
    TaskStatus,
    TaskState,
    Task,
    SendMessageRequest,
    SendMessageSuccessResponse,
    JSONRPCErrorResponse,
    TextPart,
    Part,
    Message,
    Role,
    MessageSendParams,
    MessageSendConfiguration,
    GetTaskRequest,
    TaskQueryParams,
    GetTaskSuccessResponse,
    FilePart,
)

from src.simple_pdf_processor import SimplePDFProcessor
from src.llm_reviewer import LLMReviewer
from src.paper_search import PaperSearch, PaperSearchResult
from src.review_engine import ReviewEngine, ReviewCriteria
from src.llm_client import LLMClient
from src.output_formatter import OutputFormatter
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)


class ReviewerAgent:
    """Reviewer Agent for analyzing academic papers."""

    def __init__(self):
        """Initialize the reviewer agent."""
        logger.info("Initializing ReviewerAgent")
        self.review_criteria = os.getenv("REVIEW_CRITERIA", "academic_peer_review")
        self.language = os.getenv("REVIEW_LANGUAGE", "English")
        # Initialize components
        self.pdf_processor = SimplePDFProcessor()
        self.paper_search = PaperSearch()
        self.llm_client = LLMClient()
        self.llm_reviewer = LLMReviewer(
            llm_client=self.llm_client, language=self.language
        )
        self.review_engine = ReviewEngine(llm_client=self.llm_client)
        # Task state storage
        self.tasks = {}

        # PDF cache directory
        self.cache_dir = Path("paper")
        self.cache_dir.mkdir(exist_ok=True)

    async def route_task(self, text_content: str) -> Dict:
        """
        Use LLM to determine the task type from user input.

        Args:
            text_content: User's text input

        Returns:
            Dict with task type and parameters
        """
        if not text_content:
            return {"task_type": "welcome"}

        # Create prompt for LLM
        prompt = f"""
You are a task router for an academic paper review agent. Analyze the following user request and determine which task it falls into:

USER REQUEST: "{text_content}"

Choose one of the following task types:
1. "review_paper_url" - User wants to review a paper from a URL (or multiple URLs)
2. "review_multiple_urls" - User wants to review multiple papers from multiple URLs
3. "review_template" - User wants to see the review criteria template
4. "welcome" - The user is greeting you or asking about the agent function
5. "other" - None of the above

Return your answer in JSON format:
{{
  "task_type": "review_paper_url|review_multiple_urls|review_template|welcome|other",
  "parameters": {{
    // Additional parameters based on task type
    "query": "{text_content}" // For search_paper, please summarize the content users want to search more clearly here
    "selection":  integer // For paper_selection, this is the index or title of the paper
    "url":  // For review_paper_url
    "urls": [] // For review_multiple_urls, this is an array of URLs
  }}
}}
Your response MUST be a valid JSON object and nothing else. Do not add any explanations or markdown formatting.

Here are some examples:

User Request: "你好"
{{
  "task_type": "welcome",
}}

User Request: "I want you review this paper https://arxiv.org/abs/1706.03762"
{{
  "task_type": "review_paper_url",
  "parameters": {{
    "query": "https://arxiv.org/abs/1706.03762",
    "url": "https://arxiv.org/abs/1706.03762"
  }}
}}

User Request: "Please review these papers: https://arxiv.org/abs/1706.03762 and https://arxiv.org/abs/2005.14165"
{{
  "task_type": "review_multiple_urls",
  "parameters": {{
    "query": "Please review these papers: https://arxiv.org/abs/1706.03762 and https://arxiv.org/abs/2005.14165",
    "urls": ["https://arxiv.org/abs/1706.03762", "https://arxiv.org/abs/2005.14165"]
  }}
}}

"""

        try:
            # Get response from LLM
            response = await self.llm_client.generate_text(prompt)

            # Parse JSON response
            try:
                result = json.loads(response)
                logger.info(f"Task routing result: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response}")
                # Default to search if we can't parse the response
                return {
                    "task_type": "search_paper",
                    "parameters": {"query": text_content},
                }

        except Exception as e:
            logger.error(f"Error in task routing: {e}")
            # Default to search on error
            return {"task_type": "search_paper", "parameters": {"query": text_content}}

    def _generate_cache_filename(self, paper_result: "PaperSearchResult") -> str:
        """
        Generate a cache filename for a paper.

        Args:
            paper_result: PaperSearchResult object

        Returns:
            Cache filename
        """
        # Create a unique identifier based on paper info
        title = paper_result.title or "unknown_title"
        authors = (
            "_".join(paper_result.authors[:2])
            if paper_result.authors
            else "unknown_author"
        )
        year = str(paper_result.year) if paper_result.year else "unknown_year"

        # Clean filename components
        title_clean = re.sub(r"[^\w\s-]", "", title).strip()[:50]
        title_clean = re.sub(r"[-\s]+", "_", title_clean)
        authors_clean = re.sub(r"[^\w\s-]", "", authors).strip()[:30]
        authors_clean = re.sub(r"[-\s]+", "_", authors_clean)

        # Create hash for uniqueness
        content_hash = hashlib.md5(
            f"{paper_result.url}_{paper_result.pdf_url}".encode()
        ).hexdigest()[:8]

        filename = f"{title_clean}_{authors_clean}_{year}_{content_hash}.pdf"
        return filename

    def _get_cached_pdf_path(self, paper_result: "PaperSearchResult") -> Path:
        """
        Get the cache path for a paper.

        Args:
            paper_result: PaperSearchResult object

        Returns:
            Path to cached PDF
        """
        filename = self._generate_cache_filename(paper_result)
        return self.cache_dir / filename

    async def _cache_pdf(
        self, pdf_data: bytes, paper_result: "PaperSearchResult"
    ) -> Path:
        """
        Cache PDF data to disk.

        Args:
            pdf_data: PDF content as bytes
            paper_result: PaperSearchResult object

        Returns:
            Path to cached file
        """
        cache_path = self._get_cached_pdf_path(paper_result)

        try:
            with open(cache_path, "wb") as f:
                f.write(pdf_data)
            logger.info(f"Cached PDF to: {cache_path}")
            return cache_path
        except Exception as e:
            logger.error(f"Failed to cache PDF: {e}")
            raise

    async def _load_cached_pdf(
        self, paper_result: "PaperSearchResult"
    ) -> Optional[bytes]:
        """
        Load PDF from cache if it exists.

        Args:
            paper_result: PaperSearchResult object

        Returns:
            PDF content as bytes or None if not cached
        """
        cache_path = self._get_cached_pdf_path(paper_result)

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    pdf_data = f.read()
                logger.info(f"Loaded PDF from cache: {cache_path}")
                return pdf_data
            except Exception as e:
                logger.error(f"Failed to load cached PDF: {e}")
                return None

        return None

    async def process_pdf(self, pdf_data: bytes) -> Tuple[str, Dict]:
        """Process a PDF file."""
        logger.info("Processing PDF file")
        return self.pdf_processor.extract_text_simple(pdf_data)

    async def search_papers(
        self, query: str, max_results: int = 5
    ) -> List[PaperSearchResult]:
        """Search for papers."""
        logger.info(f"Searching for papers with query: {query}")
        return await self.paper_search.search_papers(query, max_results)

    async def review_paper(self, paper_text: str, paper_info: Dict) -> Dict:
        """Review a paper."""
        logger.info(f"Reviewing paper: {paper_info.get('title', 'Unknown Title')}")
        review_result = await self.review_engine.analyze_paper(paper_text, paper_info)
        return review_result.to_dict()

    async def handle_pdf_review(self, pdf_data: bytes, task_id: str) -> str:
        """Handle PDF review request."""
        logger.info(f"Handling PDF review for task: {task_id}")

        # Extract text from PDF
        paper_text, metadata = await self.process_pdf(pdf_data)
        print("------------------------------")
        print(paper_text)
        print(metadata)
        print("------------------------------")

        # Extract paper info using simplified method
        paper_info = self.pdf_processor.extract_paper_info_simple(paper_text, metadata)

        # Generate LLM-powered review
        formatted_review = await self.llm_reviewer.generate_review(
            paper_info, paper_text
        )

        return formatted_review

    async def handle_search_review(self, query: str, context_id: str) -> str:
        """Handle search and review request."""
        logger.info(f"Handling search and review for context: {context_id}")

        # Search for papers
        search_results = await self.search_papers(query)

        if not search_results:
            return "No papers found matching your query. Please try a different search term."

        # Store search results in task state
        self.tasks[context_id] = {
            "state": "search_results",
            "search_results": [result.to_dict() for result in search_results],
        }

        # Format search results for user confirmation
        formatted_results = OutputFormatter.format_search_results_for_confirmation(
            [result.to_dict() for result in search_results]
        )

        return formatted_results

    async def handle_paper_selection(self, selection: str, context_id: str) -> str:
        """Handle paper selection for review."""
        logger.info(f"Handling paper selection for context: {context_id}")

        # Check if task exists
        if (
            context_id not in self.tasks
            or self.tasks[context_id]["state"] != "search_results"
        ):
            return "No active search results found. Please start a new search."

        # Get search results
        search_results = self.tasks[context_id]["search_results"]

        # Parse selection
        try:
            # Try to parse as index
            index = selection - 1
            if 0 <= index < len(search_results):
                selected_paper = search_results[index]
            else:
                return f"Invalid selection. Please choose a number between 1 and {len(search_results)}."
        except ValueError:
            # Try to match by title
            matches = []
            for paper in search_results:
                if selection.lower() in paper["title"].lower():
                    matches.append(paper)

            if len(matches) == 1:
                selected_paper = matches[0]
            elif len(matches) > 1:
                return f"Multiple matches found. Please be more specific or use the paper number."
            else:
                return "No matching paper found. Please try again."

        # Update task state
        self.tasks[context_id]["state"] = "paper_selected"
        self.tasks[context_id]["selected_paper"] = selected_paper

        # Initialize variables
        paper_text = ""
        pdf_available = False

        # Check if paper has PDF URL and try to get PDF
        if selected_paper.get("pdf_url"):
            # Create PaperSearchResult object for caching
            paper_result = PaperSearchResult(**selected_paper)

            # Try to load from cache first
            pdf_data = await self._load_cached_pdf(paper_result)

            if pdf_data is None:
                # Download PDF if not in cache
                logger.info(
                    f"PDF not in cache, downloading from: {selected_paper['pdf_url']}"
                )
                pdf_data = await self.paper_search.download_paper_pdf(paper_result)

                if pdf_data:
                    # Cache the downloaded PDF
                    try:
                        await self._cache_pdf(pdf_data, paper_result)
                        pdf_available = True
                    except Exception as e:
                        logger.warning(
                            f"Failed to cache PDF, continuing with review: {e}"
                        )
                        pdf_available = True
                else:
                    logger.warning(
                        f"Could not download PDF from: {selected_paper['pdf_url']}"
                    )
                    pdf_available = False
            else:
                logger.info("Using cached PDF for review")
                pdf_available = True

            # Process PDF if available
            if pdf_available and pdf_data:
                try:
                    paper_text, metadata = await self.process_pdf(pdf_data)

                    # Update paper info with extracted metadata
                    paper_info = self.pdf_processor.extract_paper_info_simple(
                        paper_text, metadata
                    )
                    for key, value in paper_info.items():
                        if value and not selected_paper.get(key):
                            selected_paper[key] = value
                except Exception as e:
                    logger.error(f"Failed to process PDF: {e}")
                    paper_text = ""
                    pdf_available = False

        # If no PDF is available, use abstract and available information for review
        if not pdf_available or not paper_text:
            logger.info(
                "No PDF available, generating review based on abstract and metadata"
            )
            # Use abstract as paper text for review
            paper_text = selected_paper.get("abstract", "")
            if not paper_text:
                paper_text = f"Title: {selected_paper.get('title', 'Unknown')}\nAuthors: {', '.join(selected_paper.get('authors', ['Unknown']))}"

        # Generate comprehensive review using ReviewEngine (preferred method)
        try:
            # Use ReviewEngine for structured analysis
            review_result = await self.review_engine.analyze_paper(
                paper_text, selected_paper
            )

            # Format the review result using the existing to_dict method
            review_dict = review_result.to_dict()

            # Create a comprehensive formatted review
            formatted_review = self._format_comprehensive_review(review_dict)

            # Add note about PDF availability
            if not pdf_available:
                formatted_review += "\n\n**注意**: 此审稿基于论文摘要和元数据生成，因为PDF文件不可用或下载失败。"

            return formatted_review

        except Exception as e:
            logger.error(f"ReviewEngine failed, falling back to LLMReviewer: {e}")
            # Fallback to LLMReviewer
            try:
                formatted_review = await self.llm_reviewer.generate_review(
                    selected_paper, paper_text
                )

                # Add note about PDF availability
                if not pdf_available:
                    formatted_review += "\n\n**注意**: 此审稿基于论文摘要和元数据生成，因为PDF文件不可用或下载失败。"

                return formatted_review
            except Exception as e2:
                logger.error(f"LLMReviewer also failed: {e2}")
                # Last resort: return basic information with error
                return f"""# 论文信息

## 基本信息
- **标题**: {selected_paper.get("title", "Unknown Title")}
- **作者**: {", ".join(selected_paper.get("authors", ["Unknown Author"]))}
- **年份**: {selected_paper.get("year", "Unknown Year")}
- **来源**: {selected_paper.get("source", "Unknown Source")}

## 摘要
{selected_paper.get("abstract", "No abstract available")}

## 错误信息
审稿生成失败。错误: {str(e2)}

**注意**: 由于PDF不可用且审稿系统出现错误，只能提供基本论文信息。
"""

    def _format_comprehensive_review(self, review_dict: Dict) -> str:
        """
        Format a comprehensive review from ReviewEngine result.

        Args:
            review_dict: Dictionary from ReviewResult.to_dict()

        Returns:
            Formatted review string
        """
        paper_info = review_dict.get("paper_info", {})
        scores = review_dict.get("review_scores", {})
        detailed_review = review_dict.get("detailed_review", {})

        # Build formatted review
        formatted_review = f"""# 论文审稿报告

## 基本信息
- **标题**: {paper_info.get("title", "Unknown Title")}
- **作者**: {", ".join(paper_info.get("authors", ["Unknown Author"]))}
- **年份**: {paper_info.get("year", "Unknown Year")}
- **来源**: {paper_info.get("source", "Unknown Source")}

## 评分总览
"""

        # Add scores
        for criterion, score in scores.items():
            if criterion != "overall":
                formatted_review += (
                    f"- **{criterion.replace('_', ' ').title()}**: {score}/10\n"
                )

        overall_score = scores.get("overall", 0)
        formatted_review += f"- **总体评分**: {overall_score}/10\n\n"

        # Add detailed review sections
        summary = detailed_review.get("summary", "No summary available")
        formatted_review += f"## 摘要\n{summary}\n\n"

        strengths = detailed_review.get("strengths", [])
        formatted_review += "## 优点\n"
        for i, strength in enumerate(strengths, 1):
            formatted_review += f"{i}. {strength}\n"
        formatted_review += "\n"

        weaknesses = detailed_review.get("weaknesses", [])
        formatted_review += "## 缺点\n"
        for i, weakness in enumerate(weaknesses, 1):
            formatted_review += f"{i}. {weakness}\n"
        formatted_review += "\n"

        questions = detailed_review.get("questions", [])
        if questions:
            formatted_review += "## 问题\n"
            for i, question in enumerate(questions, 1):
                formatted_review += f"{i}. {question}\n"
            formatted_review += "\n"

        return formatted_review

    async def handle_multiple_paper_urls(self, urls: List[str], context_id: str) -> str:
        """
        Handle review of multiple papers from URLs.

        Args:
            urls: List of URLs to papers
            context_id: Context ID for task tracking

        Returns:
            Formatted combined review of all papers
        """
        logger.info(f"Handling multiple paper URLs review for context: {context_id}")

        # Store task state
        self.tasks[context_id] = {
            "state": "multiple_url_review",
            "urls": urls,
            "reviews": [],
            "current_index": 0,
            "total": len(urls),
        }

        all_reviews = []
        failed_urls = []

        # Process each URL
        for i, url in enumerate(urls):
            logger.info(f"Processing URL {i + 1}/{len(urls)}: {url}")

            # Update task state
            self.tasks[context_id]["current_index"] = i

            try:
                # Download PDF from URL
                pdf_data = await self.paper_search.download_paper_from_url(url)

                if pdf_data:
                    # Process PDF
                    paper_text, metadata = await self.process_pdf(pdf_data)

                    # Extract paper info
                    paper_info = self.pdf_processor.extract_paper_info_simple(
                        paper_text, metadata
                    )

                    # Add URL to paper info
                    paper_info["url"] = url

                    # Generate review
                    try:
                        # Try using ReviewEngine first
                        review_result = await self.review_engine.analyze_paper(
                            paper_text, paper_info
                        )
                        review_dict = review_result.to_dict()
                        formatted_review = self._format_comprehensive_review(
                            review_dict
                        )
                    except Exception as e:
                        logger.error(
                            f"ReviewEngine failed for URL {url}, falling back to LLMReviewer: {e}"
                        )
                        # Fallback to LLMReviewer
                        formatted_review = await self.llm_reviewer.generate_review(
                            paper_info, paper_text
                        )

                    # Store review
                    all_reviews.append(
                        {
                            "url": url,
                            "review": formatted_review,
                            "paper_info": paper_info,
                        }
                    )
                    self.tasks[context_id]["reviews"].append(
                        {
                            "url": url,
                            "review": formatted_review,
                            "paper_info": paper_info,
                        }
                    )
                else:
                    logger.warning(f"Could not download PDF from URL: {url}")
                    failed_urls.append(url)
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                failed_urls.append(url)

        # Combine all reviews
        combined_review = self._format_multiple_reviews(all_reviews, failed_urls)

        return combined_review

    def _format_multiple_reviews(
        self, reviews: List[Dict], failed_urls: List[str]
    ) -> str:
        """
        Format multiple reviews into a single response.

        Args:
            reviews: List of review dictionaries
            failed_urls: List of URLs that failed to process

        Returns:
            Formatted combined review
        """
        if not reviews:
            if failed_urls:
                return (
                    f"无法处理提供的URL。请检查URL是否有效，并确保它们指向可下载的PDF文件。\n\n失败的URL:\n"
                    + "\n".join([f"- {url}" for url in failed_urls])
                )
            else:
                return "未提供有效的URL。请提供指向学术论文的URL。"

        # Start with a header
        combined = f"# 多篇论文审稿报告\n\n共处理 {len(reviews)} 篇论文"
        if failed_urls:
            combined += f"，{len(failed_urls)} 篇处理失败"
        combined += "。\n\n"

        # Add table of contents
        combined += "## 目录\n\n"
        for i, review_data in enumerate(reviews, 1):
            paper_info = review_data.get("paper_info", {})
            title = paper_info.get("title", f"论文 {i}")
            combined += f"{i}. [{title}](#paper-{i})\n"
        combined += "\n"

        # Add each review
        for i, review_data in enumerate(reviews, 1):
            url = review_data.get("url", "")
            review = review_data.get("review", "")

            # Add anchor and URL
            combined += f"<a name='paper-{i}'></a>\n\n"
            combined += f"# {i}. 论文审稿\n\n"
            combined += f"**URL**: {url}\n\n"

            # Add the review content, but remove the first heading line since we've added our own
            review_lines = review.split("\n")
            if review_lines and review_lines[0].startswith("# "):
                review = "\n".join(review_lines[1:])

            combined += review

            # Add separator between reviews
            if i < len(reviews):
                combined += "\n\n---\n\n"

        # Add failed URLs if any
        if failed_urls:
            combined += "\n\n## 处理失败的URL\n\n"
            for url in failed_urls:
                combined += f"- {url}\n"
            combined += "\n处理失败可能是由于URL无效、PDF下载失败或处理过程中出错。"

        return combined

    async def handle_review_template(self) -> str:
        """Return the review criteria template."""
        logger.info("Handling review template request")

        # Format review criteria
        formatted_template = OutputFormatter.format_review_criteria(
            self.review_criteria
        )

        return formatted_template


class ReviewerAgentExecutor(AgentExecutor):
    """Reviewer Agent Executor for A2A protocol integration."""

    def __init__(self):
        """Initialize the ReviewerAgentExecutor."""
        logger.info("Initializing ReviewerAgentExecutor")
        self.agent = ReviewerAgent()
        self.task_states = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute the agent with the given context and event queue.

        Args:
            context: The request context
            event_queue: The event queue for sending messages
        """
        logger.info("Executing ReviewerAgentExecutor")

        # Extract message from context
        msg = context.message
        text_content = []
        pdf_data = None
        pdf_filename = None
        context_id = context.context_id

        # Process message parts
        if msg and hasattr(msg, "parts"):
            for part in msg.parts:
                # Handle text parts
                if hasattr(part.root, "kind") and part.root.kind == "text":
                    text_content.append(part.root.text)
                    logger.info(f"Received text message: {part.root.text}")

                # Handle file parts (PDF uploads)
                elif hasattr(part.root, "kind") and part.root.kind == "file":
                    if hasattr(part.root, "file"):
                        file_data = part.root.file
                        file_name = file_data.name
                        file_bytes = base64.b64decode(file_data.bytes)

                        # Check if it's a PDF
                        if file_name.lower().endswith(".pdf"):
                            pdf_data = file_bytes
                            pdf_filename = file_name
                            logger.info(f"Received PDF file: {file_name}")

        # Store task state if it doesn't exist
        if context_id not in self.task_states:
            self.task_states[context_id] = {"state": "initial", "history": []}

        # Combine text content
        combined_text = " ".join(text_content) if text_content else ""

        # Generate response based on input
        response = ""

        try:
            # Handle PDF review if PDF was uploaded
            if pdf_data:
                # Update task state
                self.task_states[context_id]["state"] = "pdf_review"
                self.task_states[context_id]["pdf_filename"] = pdf_filename

                # Process PDF
                response = await self.agent.handle_pdf_review(pdf_data, context_id)

            # Handle text-based requests
            elif combined_text:
                # Route task
                task_info = await self.agent.route_task(combined_text)
                task_type = task_info.get("task_type", "other")

                if task_type == "welcome":
                    response = "Welcome to the Academic Paper Reviewer! I can help you provide detailed reviews for academic papers. How can I assist you today?"

                elif task_type == "search_paper":
                    query = task_info.get("parameters", {}).get("query", combined_text)
                    response = await self.agent.handle_search_review(query, context_id)

                elif task_type == "paper_selection":
                    selection = task_info.get("parameters", {}).get("selection", "")
                    response = await self.agent.handle_paper_selection(
                        selection, context_id
                    )

                elif task_type == "review_template":
                    response = await self.agent.handle_review_template()

                elif task_type == "review_paper_url":
                    url = task_info.get("parameters", {}).get("url", "")
                    if url:
                        # Download PDF from URL
                        pdf_data = (
                            await self.agent.paper_search.download_paper_from_url(url)
                        )
                        if pdf_data:
                            response = await self.agent.handle_pdf_review(
                                pdf_data, context_id
                            )
                        else:
                            response = "Could not download the paper from the provided URL. Please check the URL and try again."
                    else:
                        response = "Please provide a valid URL to the paper you want to review."

                elif task_type == "review_multiple_urls":
                    urls = task_info.get("parameters", {}).get("urls", [])
                    if urls:
                        response = await self.agent.handle_multiple_paper_urls(
                            urls, context_id
                        )
                    else:
                        response = "Please provide valid URLs to the papers you want to review."

                else:
                    response = "I'm not sure how to help with that request. I just can review papers if you provide a PDF or URL."

            # Default welcome message if no input
            else:
                response = "Welcome to the Academic Paper Reviewer! I can help you search for papers and provide detailed reviews. How can I assist you today?"

            # Send response
            await event_queue.enqueue_event(new_agent_text_message(response))

        except Exception as e:
            logger.error(f"Error in ReviewerAgentExecutor.execute: {e}")
            error_message = f"An error occurred while processing your request: {str(e)}"
            await event_queue.enqueue_event(new_agent_text_message(error_message))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Cancel the current execution.

        Args:
            context: The request context
            event_queue: The event queue for sending messages
        """
        logger.info(f"Cancelling task: {context.task_id}")

        # Clean up any resources if needed
        if context.task_id in self.task_states:
            # Mark task as cancelled
            self.task_states[context.task_id]["state"] = "cancelled"

        # Send cancellation message
        await event_queue.enqueue_event(new_agent_text_message("Task cancelled."))
