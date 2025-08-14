"""
PDF Processor module for extracting text from PDF files.
"""
import io
import logging
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import pdfplumber

# Configure logging
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Class for extracting text and metadata from PDF files.
    Uses both PyPDF2 and pdfplumber for robust extraction.
    """
    
    def __init__(self):
        """Initialize the PDF processor."""
        logger.info("Initializing PDF processor")
    
    def extract_text(self, pdf_data: Union[bytes, str], use_pdfplumber: bool = True) -> Tuple[str, Dict]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_data: PDF file content as bytes or file path as string
            use_pdfplumber: Whether to use pdfplumber for text extraction (more accurate but slower)
                           If False, falls back to PyPDF2 (faster but less accurate)
        
        Returns:
            Tuple containing:
                - Extracted text as string
                - Metadata dictionary
        """
        try:
            if isinstance(pdf_data, str):
                # pdf_data is a file path
                logger.info(f"Extracting text from PDF file: {pdf_data}")
                return self._extract_from_file(pdf_data, use_pdfplumber)
            else:
                # pdf_data is bytes
                logger.info("Extracting text from PDF bytes")
                return self._extract_from_bytes(pdf_data, use_pdfplumber)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _extract_from_file(self, file_path: str, use_pdfplumber: bool) -> Tuple[str, Dict]:
        """Extract text from a PDF file path."""
        if use_pdfplumber:
            return self._extract_with_pdfplumber_file(file_path)
        else:
            return self._extract_with_pypdf2_file(file_path)
    
    def _extract_from_bytes(self, pdf_bytes: bytes, use_pdfplumber: bool) -> Tuple[str, Dict]:
        """Extract text from PDF bytes."""
        if use_pdfplumber:
            return self._extract_with_pdfplumber_bytes(pdf_bytes)
        else:
            return self._extract_with_pypdf2_bytes(pdf_bytes)
    
    def _extract_with_pypdf2_file(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text using PyPDF2 from a file path."""
        with open(file_path, 'rb') as file:
            return self._extract_with_pypdf2_fileobj(file)
    
    def _extract_with_pypdf2_bytes(self, pdf_bytes: bytes) -> Tuple[str, Dict]:
        """Extract text using PyPDF2 from bytes."""
        with io.BytesIO(pdf_bytes) as file:
            return self._extract_with_pypdf2_fileobj(file)
    
    def _extract_with_pypdf2_fileobj(self, file_obj) -> Tuple[str, Dict]:
        """Extract text using PyPDF2 from a file object."""
        reader = PyPDF2.PdfReader(file_obj)
        metadata = reader.metadata
        text = ""
        
        # Extract text from each page
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n\n"
        
        # Convert metadata to dict if it's not None
        metadata_dict = {}
        if metadata:
            for key in metadata:
                if metadata[key]:
                    # Remove the leading slash from keys if present
                    clean_key = key[1:] if key.startswith('/') else key
                    metadata_dict[clean_key] = metadata[key]
        
        return text, metadata_dict
    
    def _extract_with_pdfplumber_file(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text using pdfplumber from a file path."""
        with pdfplumber.open(file_path) as pdf:
            return self._extract_with_pdfplumber_pdf(pdf)
    
    def _extract_with_pdfplumber_bytes(self, pdf_bytes: bytes) -> Tuple[str, Dict]:
        """Extract text using pdfplumber from bytes."""
        with io.BytesIO(pdf_bytes) as file:
            with pdfplumber.open(file) as pdf:
                return self._extract_with_pdfplumber_pdf(pdf)
    
    def _extract_with_pdfplumber_pdf(self, pdf) -> Tuple[str, Dict]:
        """Extract text using pdfplumber from a pdfplumber PDF object."""
        text = ""
        metadata = pdf.metadata
        
        # Extract text from each page
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        # Convert metadata to dict if it's not None
        metadata_dict = {}
        if metadata:
            metadata_dict = {k: v for k, v in metadata.items() if v}
        
        return text, metadata_dict
    
    def extract_paper_info(self, text: str, metadata: Dict) -> Dict:
        """
        Extract paper information from text and metadata.
        
        Args:
            text: Extracted text from PDF
            metadata: Metadata dictionary from PDF
        
        Returns:
            Dictionary containing paper information (title, authors, abstract, etc.)
        """
        paper_info = {
            "title": self._extract_title(text, metadata),
            "authors": self._extract_authors(text, metadata),
            "abstract": self._extract_abstract(text),
            "keywords": self._extract_keywords(text),
            "year": self._extract_year(metadata),
        }
        return paper_info
    
    def _extract_title(self, text: str, metadata: Dict) -> str:
        """Extract paper title from text and metadata."""
        # First try to get title from metadata
        if metadata and "Title" in metadata:
            return metadata["Title"]
        
        # Otherwise try to extract from first page
        lines = text.split('\n')
        if lines:
            # Assume the first non-empty line is the title
            for line in lines:
                if line.strip():
                    return line.strip()
        
        return "Unknown Title"
    
    def _extract_authors(self, text: str, metadata: Dict) -> List[str]:
        """Extract paper authors from text and metadata."""
        # First try to get authors from metadata
        if metadata and "Author" in metadata:
            # Split authors by comma or 'and'
            author_text = metadata["Author"]
            return [a.strip() for a in author_text.replace(" and ", ", ").split(",")]
        
        # TODO: Implement more sophisticated author extraction from text
        return []
    
    def _extract_abstract(self, text: str) -> str:
        """Extract paper abstract from text."""
        # Look for common abstract indicators
        abstract = ""
        lower_text = text.lower()
        
        # Try to find abstract section
        abstract_start = lower_text.find("abstract")
        if abstract_start != -1:
            # Find the end of the abstract section
            possible_ends = [
                lower_text.find("introduction"),
                lower_text.find("keywords"),
                lower_text.find("1 introduction"),
                lower_text.find("i. introduction"),
            ]
            
            # Filter out -1 values and find the closest end
            valid_ends = [end for end in possible_ends if end > abstract_start]
            if valid_ends:
                abstract_end = min(valid_ends)
                abstract = text[abstract_start:abstract_end].strip()
                # Remove the "abstract" header
                if abstract.lower().startswith("abstract"):
                    abstract = abstract[8:].strip()
            
        return abstract
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        keywords = []
        lower_text = text.lower()
        
        # Look for keywords section
        keyword_indicators = ["keywords:", "keywords", "key words:", "key words"]
        for indicator in keyword_indicators:
            start_idx = lower_text.find(indicator)
            if start_idx != -1:
                # Find the end of the keywords section
                end_idx = lower_text.find("\n\n", start_idx)
                if end_idx == -1:  # If no double newline, try single newline
                    end_idx = lower_text.find("\n", start_idx)
                
                if end_idx != -1:
                    keyword_text = text[start_idx + len(indicator):end_idx].strip()
                    # Split by common separators
                    for sep in [";", ",", "•"]:
                        if sep in keyword_text:
                            keywords = [k.strip() for k in keyword_text.split(sep)]
                            break
                break
        
        return keywords
    
    def _extract_year(self, metadata: Dict) -> Optional[int]:
        """Extract publication year from metadata."""
        if not metadata:
            return None
        
        # Try different metadata fields
        for field in ["CreationDate", "ModDate", "Date"]:
            if field in metadata:
                date_str = metadata[field]
                # Try to extract year from date string
                import re
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    try:
                        return int(year_match.group(1))
                    except ValueError:
                        pass
        
        return None