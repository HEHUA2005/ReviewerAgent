"""
Output Formatter module for formatting review results.
"""
import json
import logging
from typing import Dict, List, Optional, Union

from src.review_engine import ReviewResult

# Configure logging
logger = logging.getLogger(__name__)

class OutputFormatter:
    """Class for formatting review results."""
    
    @staticmethod
    def format_review(review_result: ReviewResult, output_format: str = "json") -> str:
        """
        Format a review result.
        
        Args:
            review_result: ReviewResult object
            output_format: Output format ("json", "text", "markdown")
        
        Returns:
            Formatted review
        """
        if output_format == "json":
            return OutputFormatter.format_as_json(review_result)
        elif output_format == "text":
            return OutputFormatter.format_as_text(review_result)
        elif output_format == "markdown":
            return OutputFormatter.format_as_markdown(review_result)
        else:
            logger.warning(f"Unknown output format: {output_format}, using json")
            return OutputFormatter.format_as_json(review_result)
    
    @staticmethod
    def format_as_json(review_result: ReviewResult) -> str:
        """Format review result as JSON."""
        return review_result.to_json()
    
    @staticmethod
    def format_as_text(review_result: ReviewResult) -> str:
        """Format review result as plain text."""
        return str(review_result)
    
    @staticmethod
    def format_as_markdown(review_result: ReviewResult) -> str:
        """Format review result as Markdown."""
        paper_info = review_result.paper_info
        scores = review_result.scores
        
        # Paper info
        md = f"# Review of: {paper_info.get('title', 'Unknown Title')}\n\n"
        
        if paper_info.get('authors'):
            md += f"**Authors**: {', '.join(paper_info['authors'])}\n"
        
        if paper_info.get('year'):
            md += f"**Year**: {paper_info['year']}\n"
        
        if paper_info.get('source'):
            md += f"**Source**: {paper_info['source']}\n"
        
        md += "\n## Review Scores\n\n"
        
        # Scores
        for key, score in scores.items():
            md += f"- **{key.title()}**: {score}/10\n"
        md += f"- **Overall**: {round(review_result.overall_score, 1)}/10\n\n"
        
        # Summary
        md += f"## Summary\n\n{review_result.summary}\n\n"
        
        # Strengths
        md += "## Strengths\n\n"
        for strength in review_result.strengths:
            md += f"- {strength}\n"
        md += "\n"
        
        # Weaknesses
        md += "## Weaknesses\n\n"
        for weakness in review_result.weaknesses:
            md += f"- {weakness}\n"
        md += "\n"
        
        # Questions (if any)
        if review_result.questions:
            md += "## Questions\n\n"
            for question in review_result.questions:
                md += f"- {question}\n"
        
        return md
    
    @staticmethod
    def format_search_results_for_confirmation(results: List[Dict], output_format: str = "markdown") -> str:
        """
        Format search results for user confirmation.
        
        Args:
            results: List of search result dictionaries
            output_format: Output format ("markdown", "text")
        
        Returns:
            Formatted search results
        """
        if not results:
            return "No papers found matching your query."
        
        if output_format == "markdown":
            return OutputFormatter._format_search_results_markdown(results)
        else:
            return OutputFormatter._format_search_results_text(results)
    
    @staticmethod
    def _format_search_results_markdown(results: List[Dict]) -> str:
        """Format search results as Markdown."""
        md = "# Search Results\n\n"
        md += "I found the following papers that match your query:\n\n"
        
        for i, result in enumerate(results):
            authors_str = ", ".join(result.get("authors", [])[:3])
            if len(result.get("authors", [])) > 3:
                authors_str += " et al."
            
            year_str = f" ({result.get('year')})" if result.get("year") else ""
            
            md += f"## {i+1}. {result.get('title', 'Unknown Title')}\n\n"
            md += f"**Authors**: {authors_str}{year_str}\n\n"
            md += f"**Source**: {result.get('source', 'Unknown').title()}\n\n"
            
            if result.get("abstract"):
                # Truncate abstract if too long
                abstract = result["abstract"]
                if len(abstract) > 300:
                    abstract = abstract[:300] + "..."
                md += f"**Abstract**: {abstract}\n\n"
            
            if result.get("url"):
                md += f"**URL**: {result['url']}\n\n"
            
            if result.get("pdf_url"):
                md += f"**PDF**: {result['pdf_url']}\n\n"
            
            md += "---\n\n"
        
        md += "Please select a paper by number to review, or provide more details to refine the search."
        
        return md
    
    @staticmethod
    def _format_search_results_text(results: List[Dict]) -> str:
        """Format search results as plain text."""
        text = "Search Results\n\n"
        text += "I found the following papers that match your query:\n\n"
        
        for i, result in enumerate(results):
            authors_str = ", ".join(result.get("authors", [])[:3])
            if len(result.get("authors", [])) > 3:
                authors_str += " et al."
            
            year_str = f" ({result.get('year')})" if result.get("year") else ""
            
            text += f"{i+1}. {result.get('title', 'Unknown Title')}\n"
            text += f"   Authors: {authors_str}{year_str}\n"
            text += f"   Source: {result.get('source', 'Unknown').title()}\n"
            
            if result.get("abstract"):
                # Truncate abstract if too long
                abstract = result["abstract"]
                if len(abstract) > 200:
                    abstract = abstract[:200] + "..."
                text += f"   Abstract: {abstract}\n"
            
            text += "\n"
        
        text += "Please select a paper by number to review, or provide more details to refine the search."
        
        return text