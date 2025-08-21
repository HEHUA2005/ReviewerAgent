"""
LLM驱动的审稿模块
"""

import json
import logging
import os
from dotenv import load_dotenv, find_dotenv
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMReviewer:
    """LLM驱动的审稿器"""

    def __init__(self, llm_client=None, language="English"):
        """
        初始化LLM审稿器

        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client
        self.language = language
        logger.info(
            f"Initialize the LLMReviewer, the language is set as: {self.language}"
        )

    async def generate_review(self, paper_info: Dict, text: str) -> str:
        """
        生成审稿报告

        Args:
            paper_info: 论文信息
            text: 论文文本

        Returns:
            格式化的审稿结果
        """
        if not self.llm_client:
            logger.warning("LLM客户端不可用，使用简单审稿")
            return self._generate_simple_review(paper_info, text)

        try:
            # 使用LLM生成审稿
            return await self._generate_llm_review(paper_info, text)
        except Exception as e:
            logger.error(f"LLM审稿失败: {e}")
            # 回退到简单审稿
            return self._generate_simple_review(paper_info, text)

    async def _generate_llm_review(self, paper_info: Dict, text: str) -> str:
        """
        使用LLM生成审稿结果

        Args:
            paper_info: 论文信息
            text: 论文文本

        Returns:
            格式化的审稿结果
        """
        title = paper_info.get("title", "Unknown Title")
        authors = paper_info.get("authors", ["Unknown Author"])
        year = paper_info.get("year", "Unknown Year")

        # 限制文本长度以适应LLM token限制
        max_text_length = 30000
        truncated_text = text[:max_text_length] if len(text) > max_text_length else text

        # 构建LLM提示
        prompt = f"""You are an expert academic reviewer. Please provide a comprehensive review of the following paper.

Paper Information:
- Title: {title}
- Authors: {", ".join(authors)}
- Year: {year}

Paper Content:
{truncated_text}

Please provide your review in the following JSON format:
{{
    "summary": "A concise summary of the paper ",
    "strengths": ["Strength 1", "Strength 2", "Strength 3",...],
    "weaknesses": ["Weakness 1", "Weakness 2", "Weakness 3",...],
    "questions": ["Question 1", "Question 2",...]
}}
And for the content of the "summary", "strengths" , "weaknesses", and "questions" sections, 
please using {self.language}, but ensure other parts remain in English.
Focus on:
- Technical soundness and methodology
- Novelty and significance of contributions
- Clarity of presentation
- Experimental validation (if applicable)
- Related work coverage

The "questions" field is optional - only include it if you have specific questions for the authors.
Your response must be valid JSON only, no additional text or formatting."""

        try:
            # 调用LLM生成审稿
            response = await self.llm_client.generate_text(prompt)

            # 解析LLM响应
            try:
                # Clean response - remove markdown code blocks if present
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()

                review_data = json.loads(cleaned_response)
                return self._format_llm_review(paper_info, review_data)
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse LLM response as JSON: {response[:200]}..."
                )
                # 尝试从文本中提取信息
                return self._extract_review_from_text(paper_info, response)

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

    def _format_llm_review(self, paper_info: Dict, review_data: Dict) -> str:
        """
        格式化LLM审稿结果

        Args:
            paper_info: 论文信息
            review_data: LLM返回的审稿数据

        Returns:
            格式化的Markdown审稿报告
        """
        title = paper_info.get("title", "Unknown Title")
        authors = paper_info.get("authors", ["Unknown Author"])
        year = paper_info.get("year", "Unknown Year")

        # 提取审稿内容
        summary = review_data.get("summary", "No summary provided")
        strengths = review_data.get("strengths", ["No strengths identified"])
        weaknesses = review_data.get("weaknesses", ["No weaknesses identified"])
        questions = review_data.get("questions", [])

        # 构建Markdown格式的审稿报告
        review = f"""# 论文审稿报告

## 基本信息
- **标题**: {title}
- **作者**: {", ".join(authors)}
- **年份**: {year}

## Summary
{summary}

## Strengths
"""

        for i, strength in enumerate(strengths, 1):
            review += f"{i}. {strength}\n"

        review += f"""
## Weaknesses
"""

        for i, weakness in enumerate(weaknesses, 1):
            review += f"{i}. {weakness}\n"

        # 添加问题部分（如果有）
        if questions:
            review += f"""
## Questions
"""
            for i, question in enumerate(questions, 1):
                review += f"{i}. {question}\n"

        review += """
---
*This review was generated using AI assistance. Please consider it as a preliminary assessment.*
"""

        return review

    def _extract_review_from_text(self, paper_info: Dict, text: str) -> str:
        """
        从文本中提取审稿信息（当JSON解析失败时）

        Args:
            paper_info: 论文信息
            text: LLM返回的文本

        Returns:
            格式化的审稿报告
        """
        logger.info("尝试从文本中提取审稿信息")

        title = paper_info.get("title", "Unknown Title")
        authors = paper_info.get("authors", ["Unknown Author"])
        year = paper_info.get("year", "Unknown Year")

        # 尝试提取各个部分
        summary = self._extract_section(text, ["summary", "摘要"], "No summary found")
        strengths = self._extract_list_section(text, ["strengths", "优点", "strength"])
        weaknesses = self._extract_list_section(
            text, ["weaknesses", "缺点", "weakness"]
        )
        questions = self._extract_list_section(text, ["questions", "问题", "question"])

        # 构建审稿报告
        review = f"""# 论文审稿报告

## 基本信息
- **标题**: {title}
- **作者**: {", ".join(authors)}
- **年份**: {year}

## Summary
{summary}

## Strengths
"""

        if strengths:
            for i, strength in enumerate(strengths, 1):
                review += f"{i}. {strength}\n"
        else:
            review += "1. No specific strengths identified from the response\n"

        review += f"""
## Weaknesses
"""

        if weaknesses:
            for i, weakness in enumerate(weaknesses, 1):
                review += f"{i}. {weakness}\n"
        else:
            review += "1. No specific weaknesses identified from the response\n"

        # 添加问题部分（如果有）
        if questions:
            review += f"""
## Questions
"""
            for i, question in enumerate(questions, 1):
                review += f"{i}. {question}\n"

        review += """
---
*This review was extracted from text response due to JSON parsing failure.*
"""

        return review

    def _extract_section(self, text: str, keywords: List[str], default: str) -> str:
        """
        从文本中提取特定部分

        Args:
            text: 源文本
            keywords: 关键词列表
            default: 默认值

        Returns:
            提取的内容
        """
        text_lower = text.lower()

        for keyword in keywords:
            start_idx = text_lower.find(keyword.lower())
            if start_idx != -1:
                # 找到关键词后的内容
                start_idx += len(keyword)
                # 查找下一个部分的开始
                next_sections = [
                    "strengths",
                    "weaknesses",
                    "questions",
                    "优点",
                    "缺点",
                    "问题",
                ]
                end_idx = len(text)

                for next_section in next_sections:
                    next_idx = text_lower.find(next_section.lower(), start_idx)
                    if next_idx != -1 and next_idx < end_idx:
                        end_idx = next_idx

                content = text[start_idx:end_idx].strip()
                # 清理内容
                content = content.lstrip(":：").strip()
                if content:
                    return content

        return default

    def _extract_list_section(self, text: str, keywords: List[str]) -> List[str]:
        """
        从文本中提取列表部分

        Args:
            text: 源文本
            keywords: 关键词列表

        Returns:
            提取的列表项
        """
        text_lower = text.lower()

        for keyword in keywords:
            start_idx = text_lower.find(keyword.lower())
            if start_idx != -1:
                # 找到关键词后的内容
                start_idx += len(keyword)
                # 查找下一个部分的开始
                next_sections = [
                    "summary",
                    "strengths",
                    "weaknesses",
                    "questions",
                    "摘要",
                    "优点",
                    "缺点",
                    "问题",
                ]
                end_idx = len(text)

                for next_section in next_sections:
                    next_idx = text_lower.find(next_section.lower(), start_idx)
                    if (
                        next_idx != -1
                        and next_idx < end_idx
                        and next_section.lower() != keyword.lower()
                    ):
                        end_idx = next_idx

                content = text[start_idx:end_idx].strip()
                # 清理内容并提取列表项
                content = content.lstrip(":：").strip()

                # 尝试提取列表项
                items = []
                lines = content.split("\n")
                for line in lines:
                    line = line.strip()
                    if line and (
                        line.startswith("-")
                        or line.startswith("•")
                        or line.startswith("*")
                        or line[0].isdigit()
                        and "." in line[:3]
                    ):
                        # 清理列表标记
                        clean_line = line.lstrip("-•*").strip()
                        if clean_line.startswith(tuple("123456789")):
                            # 移除数字编号
                            clean_line = clean_line.split(".", 1)[-1].strip()
                        if clean_line:
                            items.append(clean_line)

                if items:
                    return items

        return []

    def _generate_simple_review(self, paper_info: Dict, text: str) -> str:
        """
        生成简单的审稿结果（回退方案）

        Args:
            paper_info: 论文信息
            text: 论文文本

        Returns:
            格式化的审稿结果
        """
        title = paper_info.get("title", "Unknown Title")
        authors = paper_info.get("authors", ["Unknown Author"])
        year = paper_info.get("year", "Unknown Year")
        abstract = paper_info.get("abstract", "No abstract available")

        # 简单的文本分析
        word_count = len(text.split())

        # 生成简化的审稿报告
        review = f"""# 论文审稿报告

## 基本信息
- **标题**: {title}
- **作者**: {", ".join(authors)}
- **年份**: {year}
- **字数**: {word_count} 词

## Summary
This paper titled "{title}" presents research work by {", ".join(authors)}. The paper contains approximately {word_count} words and appears to cover {abstract[:100]}{"..." if len(abstract) > 100 else ""}.

## Strengths
1. The paper appears to have a clear structure and organization
2. The topic seems relevant to the field of study
3. The authors have provided sufficient content for analysis

## Weaknesses
1. Without LLM analysis, detailed technical assessment is not available
2. Cannot evaluate the novelty and significance of contributions
3. Unable to assess experimental validation and methodology rigor

---
*This is a simplified review generated without LLM assistance. For detailed analysis, please ensure LLM client is properly configured.*
"""

        return review
