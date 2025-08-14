"""
Review Engine module for analyzing academic papers and generating reviews.
"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Union

from config import REVIEW_CRITERIA

# Configure logging
logger = logging.getLogger(__name__)

class ReviewCriteria:
    """Class representing review criteria."""
    
    ACADEMIC_PEER_REVIEW = {
        "methodology": {
            "name": "Methodology",
            "description": "Research design, experimental setup, and approach",
            "scale": {
                "1-3": "Flawed or inappropriate methods",
                "4-6": "Adequate methods with some limitations",
                "7-8": "Sound methodology with minor issues",
                "9-10": "Excellent, rigorous methodology"
            }
        },
        "novelty": {
            "name": "Novelty",
            "description": "Originality and contribution to the field",
            "scale": {
                "1-3": "Little to no novel contribution",
                "4-6": "Some novel aspects but limited impact",
                "7-8": "Significant novel contribution",
                "9-10": "Groundbreaking, highly novel work"
            }
        },
        "clarity": {
            "name": "Clarity",
            "description": "Writing quality, organization, and presentation",
            "scale": {
                "1-3": "Poor writing, difficult to understand",
                "4-6": "Adequate clarity with some issues",
                "7-8": "Well-written and clear",
                "9-10": "Exceptionally clear and well-presented"
            }
        },
        "significance": {
            "name": "Significance",
            "description": "Impact and relevance to the field",
            "scale": {
                "1-3": "Limited practical or theoretical impact",
                "4-6": "Moderate significance to the field",
                "7-8": "Important contribution with broad impact",
                "9-10": "Highly significant, field-changing work"
            }
        }
    }
    
    TECHNICAL_ANALYSIS = {
        "implementation_feasibility": {
            "name": "Implementation Feasibility",
            "description": "Ease of implementing the proposed methods",
            "scale": {
                "1-3": "Difficult or impractical to implement",
                "4-6": "Implementable with significant effort",
                "7-8": "Relatively straightforward to implement",
                "9-10": "Highly practical and easy to implement"
            }
        },
        "reproducibility": {
            "name": "Reproducibility",
            "description": "Ability to reproduce the results",
            "scale": {
                "1-3": "Insufficient details to reproduce",
                "4-6": "Partially reproducible with effort",
                "7-8": "Mostly reproducible with minor gaps",
                "9-10": "Fully reproducible with clear instructions"
            }
        },
        "code_quality": {
            "name": "Code Quality",
            "description": "Quality of provided code or algorithms",
            "scale": {
                "1-3": "Poor code quality or no code provided",
                "4-6": "Adequate code with some issues",
                "7-8": "Good code quality with minor issues",
                "9-10": "Excellent, well-documented code"
            }
        },
        "scalability": {
            "name": "Scalability",
            "description": "Ability to scale to larger datasets or problems",
            "scale": {
                "1-3": "Poor scalability, limited to small problems",
                "4-6": "Moderate scalability with limitations",
                "7-8": "Good scalability with minor concerns",
                "9-10": "Excellent scalability to large problems"
            }
        }
    }
    
    @classmethod
    def get_criteria(cls, criteria_type: str = REVIEW_CRITERIA) -> Dict:
        """Get review criteria by type."""
        if criteria_type == "academic_peer_review":
            return cls.ACADEMIC_PEER_REVIEW
        elif criteria_type == "technical_analysis":
            return cls.TECHNICAL_ANALYSIS
        else:
            logger.warning(f"Unknown criteria type: {criteria_type}, using academic_peer_review")
            return cls.ACADEMIC_PEER_REVIEW
    
    @classmethod
    def get_criteria_names(cls, criteria_type: str = REVIEW_CRITERIA) -> List[str]:
        """Get list of criteria names."""
        criteria = cls.get_criteria(criteria_type)
        return list(criteria.keys())
    
    @classmethod
    def get_criteria_descriptions(cls, criteria_type: str = REVIEW_CRITERIA) -> Dict[str, str]:
        """Get descriptions for each criterion."""
        criteria = cls.get_criteria(criteria_type)
        return {key: value["description"] for key, value in criteria.items()}
    
    @classmethod
    def format_criteria_for_prompt(cls, criteria_type: str = REVIEW_CRITERIA) -> str:
        """Format criteria for inclusion in LLM prompt."""
        criteria = cls.get_criteria(criteria_type)
        formatted = "Review Criteria:\n\n"
        
        for key, value in criteria.items():
            formatted += f"{value['name']} (1-10): {value['description']}\n"
            formatted += "Scale:\n"
            for scale, desc in value["scale"].items():
                formatted += f"  - {scale}: {desc}\n"
            formatted += "\n"
        
        return formatted


class ReviewResult:
    """Class representing a paper review result."""
    
    def __init__(
        self,
        paper_info: Dict,
        scores: Dict[str, int],
        strengths: List[str],
        weaknesses: List[str],
        questions: List[str],
        summary: str,
        criteria_type: str = REVIEW_CRITERIA,
    ):
        self.paper_info = paper_info
        self.scores = scores
        self.strengths = strengths
        self.weaknesses = weaknesses
        self.questions = questions
        self.summary = summary
        self.criteria_type = criteria_type
        
        # Calculate overall score
        if scores:
            self.overall_score = sum(scores.values()) / len(scores)
        else:
            self.overall_score = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "paper_info": self.paper_info,
            "review_scores": {
                **self.scores,
                "overall": round(self.overall_score, 1)
            },
            "detailed_review": {
                "summary": self.summary,
                "strengths": self.strengths,
                "weaknesses": self.weaknesses,
                "questions": self.questions
            },
            "metadata": {
                "review_criteria": self.criteria_type,
                "reviewer_agent": "ReviewerAgent v1.0"
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self) -> str:
        """String representation."""
        criteria = ReviewCriteria.get_criteria(self.criteria_type)
        
        result = f"Review of: {self.paper_info.get('title', 'Unknown Title')}\n\n"
        
        # Scores
        result += "Scores:\n"
        for key, score in self.scores.items():
            if key in criteria:
                result += f"- {criteria[key]['name']}: {score}/10\n"
        result += f"- Overall: {round(self.overall_score, 1)}/10\n\n"
        
        # Summary
        result += f"Summary:\n{self.summary}\n\n"
        
        # Strengths
        result += "Strengths:\n"
        for strength in self.strengths:
            result += f"- {strength}\n"
        result += "\n"
        
        # Weaknesses
        result += "Weaknesses:\n"
        for weakness in self.weaknesses:
            result += f"- {weakness}\n"
        result += "\n"
        
        # Questions (if any)
        if self.questions:
            result += "Questions:\n"
            for question in self.questions:
                result += f"- {question}\n"
        
        return result


class ReviewEngine:
    """Class for analyzing papers and generating reviews."""
    
    def __init__(self, llm_client=None):
        """Initialize the review engine."""
        logger.info("Initializing review engine")
        self.llm_client = llm_client
    
    async def analyze_paper(
        self,
        paper_text: str,
        paper_info: Dict,
        criteria_type: str = REVIEW_CRITERIA
    ) -> ReviewResult:
        """
        Analyze a paper and generate a review.
        
        Args:
            paper_text: Extracted text from the paper
            paper_info: Paper metadata
            criteria_type: Type of review criteria to use
        
        Returns:
            ReviewResult object
        """
        logger.info(f"Analyzing paper: {paper_info.get('title', 'Unknown Title')}")
        
        if self.llm_client:
            # Use LLM to generate review
            return await self._generate_review_with_llm(paper_text, paper_info, criteria_type)
        else:
            # Fallback to simple analysis
            return self._generate_simple_review(paper_text, paper_info, criteria_type)
    
    async def _generate_review_with_llm(
        self,
        paper_text: str,
        paper_info: Dict,
        criteria_type: str
    ) -> ReviewResult:
        """Generate review using LLM."""
        logger.info("Generating review with LLM")
        
        # Prepare prompt
        prompt = self._create_review_prompt(paper_text, paper_info, criteria_type)
        
        try:
            # Call LLM using generate_text method
            response = await self.llm_client.generate_text(prompt)
            
            # Check if response indicates LLM client error
            if response.startswith("LLM client not available") or response.startswith("Error generating text"):
                logger.error(f"LLM client error: {response}")
                # Fallback to simple analysis with error message
                return self._generate_simple_review_with_error(
                    paper_text,
                    paper_info,
                    criteria_type,
                    f"LLM review generation failed: {response}"
                )
            
            # Parse response
            return self._parse_llm_response(response, paper_info, criteria_type)
        except Exception as e:
            logger.error(f"Error generating review with LLM: {e}")
            # Fallback to simple analysis with error message
            return self._generate_simple_review_with_error(
                paper_text,
                paper_info,
                criteria_type,
                f"Exception during review generation: {str(e)}"
            )
    
    def _create_review_prompt(self, paper_text: str, paper_info: Dict, criteria_type: str) -> str:
        """Create prompt for LLM review generation."""
        # Get criteria descriptions
        criteria_formatted = ReviewCriteria.format_criteria_for_prompt(criteria_type)
        
        # Truncate paper text if too long
        max_text_length = 15000  # Adjust based on LLM token limits
        truncated_text = paper_text[:max_text_length] if len(paper_text) > max_text_length else paper_text
        
        # Create prompt
        prompt = f"""You are an expert academic reviewer tasked with reviewing the following paper:

Title: {paper_info.get('title', 'Unknown Title')}
Authors: {', '.join(paper_info.get('authors', ['Unknown']))}
Year: {paper_info.get('year', 'Unknown')}

Please provide a comprehensive review based on the following criteria:

{criteria_formatted}

For each criterion, provide a score from 1 to 10, where 1 is the lowest and 10 is the highest.

Your review should be divided into four parts:
1. Summary: A concise summary of the paper (2-3 paragraphs)
2. Strengths: 3-5 key strengths of the paper
3. Weaknesses: 3-5 key weaknesses or limitations
4. Questions (optional): 1-3 questions for the authors that could help clarify aspects of the paper or address concerns not related to the paper's quality

Paper text:
{truncated_text}

Your review should be thorough, constructive, and fair. Format your response as JSON with the following structure:
{{
  "scores": {{
    "criterion1": score,
    "criterion2": score,
    ...
  }},
  "summary": "Your summary here",
  "strengths": ["Strength 1", "Strength 2", ...],
  "weaknesses": ["Weakness 1", "Weakness 2", ...],
  "questions": ["Question 1", "Question 2", ...]
}}

Replace "criterion1", "criterion2", etc. with the actual criteria names: {', '.join(ReviewCriteria.get_criteria_names(criteria_type))}.
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, paper_info: Dict, criteria_type: str) -> ReviewResult:
        """Parse LLM response into ReviewResult."""
        try:
            # Clean response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            # Try to parse as JSON
            data = json.loads(cleaned_response)
            
            # Extract fields
            scores = data.get("scores", {})
            summary = data.get("summary", "No summary provided.")
            strengths = data.get("strengths", ["No strengths identified."])
            weaknesses = data.get("weaknesses", ["No weaknesses identified."])
            questions = data.get("questions", [])
            
            # Validate scores
            if not scores:
                logger.warning("No scores found in LLM response")
                
            # Validate summary
            if summary == "No summary provided.":
                logger.warning("No summary found in LLM response")
                
            # Validate strengths and weaknesses
            if strengths == ["No strengths identified."]:
                logger.warning("No strengths found in LLM response")
                
            if weaknesses == ["No weaknesses identified."]:
                logger.warning("No weaknesses found in LLM response")
            
            # Create ReviewResult
            return ReviewResult(
                paper_info=paper_info,
                scores=scores,
                strengths=strengths,
                weaknesses=weaknesses,
                questions=questions,
                summary=summary,
                criteria_type=criteria_type
            )
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON. Response: {response[:100]}...")
            # Try to extract information from text
            return self._extract_review_from_text(response, paper_info, criteria_type)
    
    def _extract_review_from_text(self, text: str, paper_info: Dict, criteria_type: str) -> ReviewResult:
        """Extract review information from text response."""
        logger.info("Extracting review from text response")
        
        # Get criteria names
        criteria_names = ReviewCriteria.get_criteria_names(criteria_type)
        
        # Extract scores
        scores = {}
        for criterion in criteria_names:
            # Look for patterns like "Methodology: 8/10" or "Methodology score: 8"
            patterns = [
                rf"{criterion}:\s*(\d+)/10",
                rf"{criterion} score:\s*(\d+)",
                rf"{criterion}.*?(\d+)/10",
                rf"{criterion}.*?score.*?(\d+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        score = int(match.group(1))
                        if 1 <= score <= 10:
                            scores[criterion] = score
                            break
                    except ValueError:
                        pass
            
            # If no score found, assign a default
            if criterion not in scores:
                scores[criterion] = 5  # Default middle score
        
        # Extract summary (assume it's between "Summary:" and the next section)
        summary_match = re.search(r"Summary:(.*?)(?:Strengths:|Weaknesses:|Questions:|$)", text, re.DOTALL | re.IGNORECASE)
        summary = summary_match.group(1).strip() if summary_match else "No summary extracted."
        
        # Extract strengths
        strengths_match = re.search(r"Strengths:(.*?)(?:Weaknesses:|Questions:|$)", text, re.DOTALL | re.IGNORECASE)
        strengths_text = strengths_match.group(1).strip() if strengths_match else ""
        strengths = [s.strip() for s in re.findall(r"[-*•]\s*(.*?)(?:\n|$)", strengths_text)]
        if not strengths:
            strengths = ["No strengths extracted."]
        
        # Extract weaknesses
        weaknesses_match = re.search(r"Weaknesses:(.*?)(?:Questions:|Strengths:|$)", text, re.DOTALL | re.IGNORECASE)
        weaknesses_text = weaknesses_match.group(1).strip() if weaknesses_match else ""
        weaknesses = [w.strip() for w in re.findall(r"[-*•]\s*(.*?)(?:\n|$)", weaknesses_text)]
        if not weaknesses:
            weaknesses = ["No weaknesses extracted."]
        
        # Extract questions
        questions_match = re.search(r"Questions:(.*?)(?:Summary:|Strengths:|Weaknesses:|$)", text, re.DOTALL | re.IGNORECASE)
        questions_text = questions_match.group(1).strip() if questions_match else ""
        questions = [q.strip() for q in re.findall(r"[-*•]\s*(.*?)(?:\n|$)", questions_text)]
        # Questions are optional, so we don't need a default
        
        # Create ReviewResult
        return ReviewResult(
            paper_info=paper_info,
            scores=scores,
            strengths=strengths,
            weaknesses=weaknesses,
            questions=questions,
            summary=summary,
            criteria_type=criteria_type
        )
    
    def _generate_simple_review_with_error(
        self,
        paper_text: str,
        paper_info: Dict,
        criteria_type: str,
        error_message: str
    ) -> ReviewResult:
        """Generate a simple review with error message."""
        logger.info(f"Generating simple review with error: {error_message}")
        
        # Get criteria names
        criteria_names = ReviewCriteria.get_criteria_names(criteria_type)
        
        # Generate default scores (middle of the range)
        scores = {criterion: 5 for criterion in criteria_names}
        
        # Generate simple summary with error message
        title = paper_info.get("title", "Unknown Title")
        authors = paper_info.get("authors", ["Unknown"])
        year = paper_info.get("year", "Unknown")
        
        summary = f"ERROR: {error_message}\n\nThis is a fallback review of '{title}' by {', '.join(authors)} ({year}). "
        
        if "abstract" in paper_info and paper_info["abstract"]:
            summary += f"The paper's abstract states: {paper_info['abstract'][:300]}..."
        else:
            # Try to extract abstract from paper text
            abstract_match = re.search(r"Abstract\s*(.*?)(?:Introduction|Keywords|References)", paper_text, re.DOTALL | re.IGNORECASE)
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                summary += f"The paper appears to be about: {abstract[:300]}..."
            else:
                summary += f"The paper is {len(paper_text)} characters long."
        
        # Generate simple strengths and weaknesses with error notice
        strengths = [
            "ERROR: Automated review failed due to LLM connection issues.",
            "The paper appears to be well-structured.",
            "The topic seems relevant to the field."
        ]
        
        weaknesses = [
            "ERROR: Automated review failed due to LLM connection issues.",
            "Without deeper analysis, specific weaknesses cannot be identified.",
            "A more thorough review would require expert analysis."
        ]
        
        # Generate questions (optional)
        questions = [
            "Could you provide more details about the methodology used in this study?",
            "How does this work compare to other recent approaches in the field?"
        ]
        
        # Create ReviewResult
        return ReviewResult(
            paper_info=paper_info,
            scores=scores,
            strengths=strengths,
            weaknesses=weaknesses,
            questions=questions,
            summary=summary,
            criteria_type=criteria_type
        )
        
    def _generate_simple_review(self, paper_text: str, paper_info: Dict, criteria_type: str) -> ReviewResult:
        """Generate a simple review without LLM."""
        logger.info("Generating simple review")
        
        # Get criteria names
        criteria_names = ReviewCriteria.get_criteria_names(criteria_type)
        
        # Generate default scores (middle of the range)
        scores = {criterion: 5 for criterion in criteria_names}
        
        # Generate simple summary
        title = paper_info.get("title", "Unknown Title")
        authors = paper_info.get("authors", ["Unknown"])
        year = paper_info.get("year", "Unknown")
        
        summary = f"This is a simple review of '{title}' by {', '.join(authors)} ({year}). "
        
        if "abstract" in paper_info and paper_info["abstract"]:
            summary += f"The paper's abstract states: {paper_info['abstract'][:300]}..."
        else:
            # Try to extract abstract from paper text
            abstract_match = re.search(r"Abstract\s*(.*?)(?:Introduction|Keywords|References)", paper_text, re.DOTALL | re.IGNORECASE)
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                summary += f"The paper appears to be about: {abstract[:300]}..."
            else:
                summary += f"The paper is {len(paper_text)} characters long."
        
        # Generate simple strengths and weaknesses
        strengths = [
            "The paper appears to be well-structured.",
            "The topic seems relevant to the field."
        ]
        
        weaknesses = [
            "Without deeper analysis, specific weaknesses cannot be identified.",
            "A more thorough review would require expert analysis."
        ]
        
        # Generate questions (optional)
        questions = [
            "Could you provide more details about the methodology used in this study?",
            "How does this work compare to other recent approaches in the field?"
        ]
        
        # Create ReviewResult
        return ReviewResult(
            paper_info=paper_info,
            scores=scores,
            strengths=strengths,
            weaknesses=weaknesses,
            questions=questions,
            summary=summary,
            criteria_type=criteria_type
        )