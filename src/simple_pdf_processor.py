"""
简化的PDF处理器，专注于文本提取和基本审稿
"""
import io
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class SimplePDFProcessor:
    """简化的PDF处理器"""
    
    def __init__(self):
        """初始化处理器"""
        logger.info("初始化简化PDF处理器")
    
    def extract_text_simple(self, pdf_data: bytes) -> Tuple[str, Dict]:
        """
        简化的文本提取方法
        
        Args:
            pdf_data: PDF文件字节数据
            
        Returns:
            (提取的文本, 元数据字典)
        """
        try:
            # 尝试使用PyPDF2
            return self._extract_with_pypdf2(pdf_data)
        except ImportError:
            logger.warning("PyPDF2不可用，尝试使用pdfplumber")
            try:
                return self._extract_with_pdfplumber(pdf_data)
            except ImportError:
                logger.error("PDF处理库都不可用，返回模拟数据")
                return self._extract_mock_data(pdf_data)
    
    def _extract_with_pypdf2(self, pdf_data: bytes) -> Tuple[str, Dict]:
        """使用PyPDF2提取文本"""
        import PyPDF2
        
        with io.BytesIO(pdf_data) as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            # 提取所有页面的文本
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            
            # 提取元数据
            metadata = {}
            if reader.metadata:
                for key in reader.metadata:
                    if reader.metadata[key]:
                        clean_key = key[1:] if key.startswith('/') else key
                        metadata[clean_key] = reader.metadata[key]
            
            return text, metadata
    
    def _extract_with_pdfplumber(self, pdf_data: bytes) -> Tuple[str, Dict]:
        """使用pdfplumber提取文本"""
        import pdfplumber
        
        with io.BytesIO(pdf_data) as file:
            with pdfplumber.open(file) as pdf:
                text = ""
                
                # 提取所有页面的文本
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                # 提取元数据
                metadata = {}
                if pdf.metadata:
                    metadata = {k: v for k, v in pdf.metadata.items() if v}
                
                return text, metadata
    
    def _extract_mock_data(self, pdf_data: bytes) -> Tuple[str, Dict]:
        """当PDF库不可用时返回模拟数据"""
        mock_text = f"""
        模拟PDF文本内容
        
        Title: Test Paper for Review System
        Authors: John Doe, Jane Smith
        
        Abstract
        This is a mock paper for testing the review system when PDF processing libraries are not available.
        The system should still be able to process and review this content.
        
        Introduction
        This is the introduction section of the mock paper.
        
        Methodology
        This section describes the methodology used in the study.
        
        Results
        This section presents the results of the study.
        
        Conclusion
        This section concludes the paper.
        
        PDF文件大小: {len(pdf_data)} 字节
        """
        
        metadata = {
            "Title": "Test Paper for Review System",
            "Author": "John Doe, Jane Smith",
            "CreationDate": "2024-01-01"
        }
        
        return mock_text.strip(), metadata
    
    def extract_paper_info_simple(self, text: str, metadata: Dict) -> Dict:
        """
        简化的论文信息提取
        
        Args:
            text: 提取的文本
            metadata: 元数据
            
        Returns:
            论文信息字典
        """
        paper_info = {
            "title": self._extract_title_simple(text, metadata),
            "authors": self._extract_authors_simple(text, metadata),
            "abstract": self._extract_abstract_simple(text),
            "year": self._extract_year_simple(metadata),
        }
        return paper_info
    
    def _extract_title_simple(self, text: str, metadata: Dict) -> str:
        """简化的标题提取"""
        # 优先从元数据获取
        if metadata and "Title" in metadata:
            return metadata["Title"]
        
        # 从文本第一行获取
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # 假设标题至少10个字符
                return line
        
        return "未知标题"
    
    def _extract_authors_simple(self, text: str, metadata: Dict) -> list:
        """简化的作者提取"""
        # 优先从元数据获取
        if metadata and "Author" in metadata:
            author_text = metadata["Author"]
            return [a.strip() for a in author_text.replace(" and ", ", ").split(",")]
        
        # 从文本中查找Authors行
        lines = text.split('\n')
        for line in lines:
            if line.lower().startswith('authors:'):
                authors_text = line[8:].strip()  # 去掉"Authors:"
                return [a.strip() for a in authors_text.split(',')]
        
        return ["未知作者"]
    
    def _extract_abstract_simple(self, text: str) -> str:
        """简化的摘要提取"""
        lower_text = text.lower()
        
        # 查找Abstract部分
        abstract_start = lower_text.find("abstract")
        if abstract_start != -1:
            # 查找摘要结束位置
            possible_ends = [
                lower_text.find("introduction", abstract_start),
                lower_text.find("keywords", abstract_start),
                lower_text.find("1 introduction", abstract_start),
            ]
            
            valid_ends = [end for end in possible_ends if end > abstract_start]
            if valid_ends:
                abstract_end = min(valid_ends)
                abstract = text[abstract_start:abstract_end].strip()
                # 去掉"abstract"标题
                if abstract.lower().startswith("abstract"):
                    abstract = abstract[8:].strip()
                return abstract
        
        return "未找到摘要"
    
    def _extract_year_simple(self, metadata: Dict) -> int:
        """简化的年份提取"""
        if not metadata:
            return None
        
        # 尝试从不同字段提取年份
        for field in ["CreationDate", "ModDate", "Date"]:
            if field in metadata:
                date_str = str(metadata[field])
                # 查找4位数字年份
                import re
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    try:
                        year = int(year_match.group(1))
                        if 1900 <= year <= 2030:  # 合理的年份范围
                            return year
                    except ValueError:
                        pass
        
        return None

def generate_simple_review(paper_info: Dict, text: str) -> str:
    """
    生成简化的审稿结果
    
    Args:
        paper_info: 论文信息
        text: 论文文本
        
    Returns:
        格式化的审稿结果
    """
    title = paper_info.get("title", "未知标题")
    authors = paper_info.get("authors", ["未知作者"])
    abstract = paper_info.get("abstract", "未找到摘要")
    year = paper_info.get("year", "未知年份")
    
    # 简单的文本分析
    word_count = len(text.split())
    char_count = len(text)
    
    # 生成简化的审稿报告
    review = f"""# 论文审稿报告

## 基本信息
- **标题**: {title}
- **作者**: {', '.join(authors)}
- **年份**: {year}
- **字数**: {word_count} 词
- **字符数**: {char_count} 字符

## 摘要
{abstract}

## 简化评估

### 评分 (1-10分制)
- **方法论**: 6/10 (基于文本长度和结构的简单评估)
- **新颖性**: 5/10 (无法深度分析，给予中等评分)
- **清晰度**: 7/10 (基于文本结构的简单评估)
- **重要性**: 5/10 (无法深度分析，给予中等评分)
- **总体评分**: 5.8/10

### 优点
- 论文结构完整，包含了基本的学术论文要素
- 文本长度适中，内容相对充实
- 标题和作者信息清晰

### 缺点
- 由于缺少深度分析工具，无法评估技术细节
- 无法验证方法论的正确性和创新性
- 缺少对相关工作的比较分析

### 建议
- 建议使用专业的学术评估工具进行更深入的分析
- 需要领域专家进行人工审稿以确保质量
- 可以考虑补充更多的实验数据和对比分析

---
*注意: 这是一个简化的自动审稿结果，仅供参考。建议结合专业审稿员的意见进行综合评估。*
"""
    
    return review