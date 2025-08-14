"""
Paper Search module for searching academic papers from various sources.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union

import arxiv
import httpx

from config import ARXIV_CATEGORIES, SEMANTIC_SCHOLAR_API_KEY, MAX_SEARCH_RESULTS

# Configure logging
logger = logging.getLogger(__name__)


class PaperSearchResult:
    """Class representing a paper search result."""

    def __init__(
        self,
        title: str,
        authors: List[str],
        abstract: str,
        url: str,
        pdf_url: Optional[str] = None,
        year: Optional[int] = None,
        source: str = "unknown",
        paper_id: Optional[str] = None,
        score: float = 0.0,
    ):
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.url = url
        self.pdf_url = pdf_url
        self.year = year
        self.source = source
        self.paper_id = paper_id
        self.score = score

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "year": self.year,
            "source": self.source,
            "paper_id": self.paper_id,
            "score": self.score,
        }

    def __str__(self) -> str:
        """String representation."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."

        year_str = f" ({self.year})" if self.year else ""

        return f"{self.title} by {authors_str}{year_str} [{self.source}]"


class PaperSearch:
    """Class for searching academic papers from various sources."""

    def __init__(self):
        """Initialize the paper search engine."""
        logger.info("Initializing paper search engine")
        # Enhanced HTTP client with better timeout and headers
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0, read=60.0),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/pdf,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

    async def search_papers(
        self,
        query: str,
        max_results: int = MAX_SEARCH_RESULTS,
        sources: List[str] = None,
    ) -> List[PaperSearchResult]:
        """
        Search for papers across multiple sources.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            sources: List of sources to search (default: ["arxiv", "semantic_scholar"])

        Returns:
            List of PaperSearchResult objects
        """
        if sources is None:
            sources = ["arxiv", "semantic_scholar"]

        logger.info(
            f"Searching for papers with query: '{query}' across sources: {sources}"
        )

        results = []

        # Search arXiv
        if "arxiv" in sources:
            try:
                arxiv_results = await self.search_arxiv(query, max_results)
                results.extend(arxiv_results)
                logger.info(f"Found {len(arxiv_results)} results from arXiv")
            except Exception as e:
                logger.error(f"Error searching arXiv: {e}")

        # Search Semantic Scholar
        if "semantic_scholar" in sources and SEMANTIC_SCHOLAR_API_KEY:
            try:
                ss_results = await self.search_semantic_scholar(query, max_results)
                results.extend(ss_results)
                logger.info(f"Found {len(ss_results)} results from Semantic Scholar")
            except Exception as e:
                logger.error(f"Error searching Semantic Scholar: {e}")

        # Sort results by score
        results.sort(key=lambda x: x.score, reverse=True)

        # Limit to max_results
        return results[:max_results]

    async def search_arxiv(
        self, query: str, max_results: int = 5
    ) -> List[PaperSearchResult]:
        """
        Search for papers on arXiv.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of PaperSearchResult objects
        """
        logger.info(f"Searching arXiv for: {query}")

        # Add category filter if specified
        if ARXIV_CATEGORIES:
            categories = ARXIV_CATEGORIES.split(",")
            category_filter = " OR ".join([f"cat:{cat.strip()}" for cat in categories])
            search_query = f"({query}) AND ({category_filter})"
        else:
            search_query = query

        # Create arXiv search client
        client = arxiv.Client()

        # Create search
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []

        # Execute search
        for result in client.results(search):
            # Extract year from published date
            year = result.published.year if result.published else None

            # Create PaperSearchResult
            paper_result = PaperSearchResult(
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                year=year,
                source="arxiv",
                paper_id=result.get_short_id(),
                score=0.9,  # Default score for arXiv results
            )

            results.append(paper_result)

        return results

    async def search_semantic_scholar(
        self, query: str, max_results: int = 5
    ) -> List[PaperSearchResult]:
        """
        Search for papers on Semantic Scholar.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of PaperSearchResult objects
        """
        logger.info(f"Searching Semantic Scholar for: {query}")

        if not SEMANTIC_SCHOLAR_API_KEY:
            logger.warning("Semantic Scholar API key not set, skipping search")
            return []

        # Semantic Scholar API endpoint
        api_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        # Prepare query parameters
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,url,year,venue,publicationDate,externalIds,openAccessPdf",
        }

        # Prepare headers
        headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}

        # Execute search
        try:
            response = await self.http_client.get(
                api_url, params=params, headers=headers
            )
            response.raise_for_status()
            data = response.json()

            results = []

            for paper in data.get("data", []):
                # Extract PDF URL if available
                pdf_url = None
                if "openAccessPdf" in paper and paper["openAccessPdf"]:
                    pdf_url = paper["openAccessPdf"].get("url")

                # Create PaperSearchResult
                paper_result = PaperSearchResult(
                    title=paper.get("title", "Unknown Title"),
                    authors=[
                        author.get("name", "Unknown")
                        for author in paper.get("authors", [])
                    ],
                    abstract=paper.get("abstract", ""),
                    url=paper.get("url", ""),
                    pdf_url=pdf_url,
                    year=paper.get("year"),
                    source="semantic_scholar",
                    paper_id=paper.get("paperId"),
                    score=0.8,  # Default score for Semantic Scholar results
                )

                results.append(paper_result)

            return results
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []

    async def get_paper_by_id(
        self, paper_id: str, source: str
    ) -> Optional[PaperSearchResult]:
        """
        Get paper details by ID from a specific source.

        Args:
            paper_id: Paper ID
            source: Source name ("arxiv" or "semantic_scholar")

        Returns:
            PaperSearchResult object or None if not found
        """
        if source == "arxiv":
            return await self._get_arxiv_paper(paper_id)
        elif source == "semantic_scholar":
            return await self._get_semantic_scholar_paper(paper_id)
        else:
            logger.error(f"Unknown source: {source}")
            return None

    async def _get_arxiv_paper(self, paper_id: str) -> Optional[PaperSearchResult]:
        """Get paper details from arXiv by ID."""
        logger.info(f"Getting arXiv paper with ID: {paper_id}")

        try:
            # Create arXiv search client
            client = arxiv.Client()

            # Search for paper by ID
            search = arxiv.Search(id_list=[paper_id], max_results=1)

            # Get the first result
            results = list(client.results(search))
            if not results:
                logger.warning(f"No paper found with ID: {paper_id}")
                return None

            result = results[0]

            # Extract year from published date
            year = result.published.year if result.published else None

            # Create PaperSearchResult
            return PaperSearchResult(
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                year=year,
                source="arxiv",
                paper_id=result.get_short_id(),
                score=1.0,  # Direct match
            )
        except Exception as e:
            logger.error(f"Error getting arXiv paper: {e}")
            return None

    async def _get_semantic_scholar_paper(
        self, paper_id: str
    ) -> Optional[PaperSearchResult]:
        """Get paper details from Semantic Scholar by ID."""
        logger.info(f"Getting Semantic Scholar paper with ID: {paper_id}")

        if not SEMANTIC_SCHOLAR_API_KEY:
            logger.warning("Semantic Scholar API key not set, skipping search")
            return None

        # Semantic Scholar API endpoint
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"

        # Prepare query parameters
        params = {
            "fields": "title,authors,abstract,url,year,venue,publicationDate,externalIds,openAccessPdf"
        }

        # Prepare headers
        headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}

        # Execute search
        try:
            response = await self.http_client.get(
                api_url, params=params, headers=headers
            )
            response.raise_for_status()
            paper = response.json()

            # Extract PDF URL if available
            pdf_url = None
            if "openAccessPdf" in paper and paper["openAccessPdf"]:
                pdf_url = paper["openAccessPdf"].get("url")

            # Create PaperSearchResult
            return PaperSearchResult(
                title=paper.get("title", "Unknown Title"),
                authors=[
                    author.get("name", "Unknown") for author in paper.get("authors", [])
                ],
                abstract=paper.get("abstract", ""),
                url=paper.get("url", ""),
                pdf_url=pdf_url,
                year=paper.get("year"),
                source="semantic_scholar",
                paper_id=paper.get("paperId"),
                score=1.0,  # Direct match
            )
        except Exception as e:
            logger.error(f"Error getting Semantic Scholar paper: {e}")
            return None

    async def download_paper_pdf(
        self, paper_result: PaperSearchResult, max_retries: int = 3
    ) -> Optional[bytes]:
        """
        Download paper PDF with enhanced error handling and retry logic.

        Args:
            paper_result: PaperSearchResult object
            max_retries: Maximum number of retry attempts

        Returns:
            PDF content as bytes or None if download failed
        """
        if not paper_result.pdf_url:
            logger.warning(f"No PDF URL available for paper: {paper_result.title}")
            return None

        logger.info(f"Downloading PDF from: {paper_result.pdf_url}")

        for attempt in range(max_retries + 1):
            try:
                # Add delay between retries (exponential backoff)
                if attempt > 0:
                    delay = 2 ** attempt  # 2, 4, 8 seconds
                    logger.info(f"Retrying download in {delay} seconds (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                
                response = await self.http_client.get(paper_result.pdf_url)
                response.raise_for_status()
                
                # Validate content type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and not paper_result.pdf_url.lower().endswith('.pdf'):
                    logger.warning(f"Response may not be PDF: {content_type}")
                
                # Validate content size (should be reasonable for a PDF)
                content_length = len(response.content)
                if content_length < 1024:  # Less than 1KB is suspicious
                    logger.warning(f"Downloaded content is very small ({content_length} bytes), may not be a valid PDF")
                elif content_length > 50 * 1024 * 1024:  # More than 50MB is very large
                    logger.warning(f"Downloaded content is very large ({content_length / 1024 / 1024:.1f} MB)")
                
                # Basic PDF validation - check for PDF header
                if response.content[:4] != b'%PDF':
                    logger.warning("Downloaded content does not appear to be a valid PDF (missing PDF header)")
                    if attempt < max_retries:
                        continue  # Try again
                    else:
                        logger.error("Final attempt: Content is not a valid PDF")
                        return None
                
                logger.info(f"Successfully downloaded PDF ({content_length / 1024:.1f} KB)")
                return response.content
                
            except httpx.TimeoutException as e:
                logger.warning(f"Timeout downloading PDF (attempt {attempt + 1}): {e}")
                if attempt >= max_retries:
                    logger.error(f"Failed to download PDF after {max_retries + 1} attempts due to timeout")
                    return None
                    
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error downloading PDF (attempt {attempt + 1}): {e.response.status_code} - {e}")
                # Don't retry for certain status codes
                if e.response.status_code in [401, 403, 404, 410]:
                    logger.error(f"Permanent error ({e.response.status_code}), not retrying")
                    return None
                if attempt >= max_retries:
                    logger.error(f"Failed to download PDF after {max_retries + 1} attempts due to HTTP error")
                    return None
                    
            except httpx.RequestError as e:
                logger.warning(f"Request error downloading PDF (attempt {attempt + 1}): {e}")
                if attempt >= max_retries:
                    logger.error(f"Failed to download PDF after {max_retries + 1} attempts due to request error")
                    return None
                    
            except Exception as e:
                logger.warning(f"Unexpected error downloading PDF (attempt {attempt + 1}): {e}")
                if attempt >= max_retries:
                    logger.error(f"Failed to download PDF after {max_retries + 1} attempts due to unexpected error")
                    return None

        return None

    async def download_paper_from_url(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """
        Download paper PDF from a URL with enhanced error handling.
        
        Args:
            url: URL to the paper (could be arXiv, direct PDF, etc.)
            max_retries: Maximum number of retry attempts
            
        Returns:
            PDF content as bytes or None if download failed
        """
        logger.info(f"Downloading paper from URL: {url}")
        
        # Handle arXiv URLs specially
        if "arxiv.org" in url:
            try:
                # Extract arXiv ID from URL
                import re
                arxiv_match = re.search(r'arxiv\.org/(?:abs/)?(\d+\.\d+)', url)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)
                    logger.info(f"Detected arXiv paper ID: {arxiv_id}")
                    # Get paper info from arXiv
                    paper_result = await self._get_arxiv_paper(arxiv_id)
                    if paper_result and paper_result.pdf_url:
                        return await self.download_paper_pdf(paper_result, max_retries)
                    else:
                        logger.warning(f"Could not get arXiv paper info for ID: {arxiv_id}")
                        return None
            except Exception as e:
                logger.error(f"Error processing arXiv URL: {e}")
                return None
        
        # Try direct PDF download with retry logic
        for attempt in range(max_retries + 1):
            try:
                # Add delay between retries
                if attempt > 0:
                    delay = 2 ** attempt  # 2, 4, 8 seconds
                    logger.info(f"Retrying URL download in {delay} seconds (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                
                response = await self.http_client.get(url)
                response.raise_for_status()
                
                # Check if response is PDF
                content_type = response.headers.get('content-type', '').lower()
                content_length = len(response.content)
                
                # Validate content
                if 'pdf' in content_type or url.lower().endswith('.pdf'):
                    # Basic PDF validation
                    if response.content[:4] != b'%PDF':
                        logger.warning("Downloaded content does not appear to be a valid PDF (missing PDF header)")
                        if attempt < max_retries:
                            continue  # Try again
                        else:
                            logger.error("Final attempt: Content is not a valid PDF")
                            return None
                    
                    logger.info(f"Successfully downloaded PDF from URL ({content_length / 1024:.1f} KB)")
                    return response.content
                else:
                    logger.warning(f"URL does not appear to be a PDF: {content_type}")
                    # For non-PDF content types, don't retry
                    return None
                    
            except httpx.TimeoutException as e:
                logger.warning(f"Timeout downloading from URL (attempt {attempt + 1}): {e}")
                if attempt >= max_retries:
                    logger.error(f"Failed to download from URL after {max_retries + 1} attempts due to timeout")
                    return None
                    
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error downloading from URL (attempt {attempt + 1}): {e.response.status_code} - {e}")
                # Don't retry for certain status codes
                if e.response.status_code in [401, 403, 404, 410]:
                    logger.error(f"Permanent error ({e.response.status_code}), not retrying")
                    return None
                if attempt >= max_retries:
                    logger.error(f"Failed to download from URL after {max_retries + 1} attempts due to HTTP error")
                    return None
                    
            except httpx.RequestError as e:
                logger.warning(f"Request error downloading from URL (attempt {attempt + 1}): {e}")
                if attempt >= max_retries:
                    logger.error(f"Failed to download from URL after {max_retries + 1} attempts due to request error")
                    return None
                    
            except Exception as e:
                logger.warning(f"Unexpected error downloading from URL (attempt {attempt + 1}): {e}")
                if attempt >= max_retries:
                    logger.error(f"Failed to download from URL after {max_retries + 1} attempts due to unexpected error")
                    return None

        return None

    def format_results_for_user(self, results: List[PaperSearchResult]) -> str:
        """
        Format search results for user confirmation.

        Args:
            results: List of PaperSearchResult objects

        Returns:
            Formatted string with search results
        """
        if not results:
            return "No papers found matching your query."

        formatted = "I found the following papers that match your query:\n\n"

        for i, result in enumerate(results):
            authors_str = ", ".join(result.authors[:3])
            if len(result.authors) > 3:
                authors_str += " et al."

            year_str = f" ({result.year})" if result.year else ""

            formatted += f"{i + 1}. **{result.title}**\n"
            formatted += f"   Authors: {authors_str}{year_str}\n"
            formatted += f"   Source: {result.source.title()}\n"
            if result.abstract:
                # Truncate abstract if too long
                abstract = (
                    result.abstract[:200] + "..."
                    if len(result.abstract) > 200
                    else result.abstract
                )
                formatted += f"   Abstract: {abstract}\n"
            formatted += "\n"

        formatted += "Please select a paper by number to review, or provide more details to refine the search."

        return formatted
