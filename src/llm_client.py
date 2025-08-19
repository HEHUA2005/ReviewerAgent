"""
LLM Client module for integration with OpenAI-compatible APIs.
"""

import logging
from typing import Dict, List, Optional, AsyncGenerator
from xml.parsers.expat import model
from openai import OpenAI
from dotenv import load_dotenv
import os
# Configure logging
logger = logging.getLogger(__name__)


class Message:
    """Class representing a chat message."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {"role": self.role, "content": self.content}


class LLMClient:
    """Class for interacting with OpenAI-compatible LLM APIs."""

    def __init__(self):
        """Initialize the LLM client."""
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("LLM_MODEL")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", 1024))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))
        logger.info("Initializing LLM client")
        self.client = self._create_client()
        if not self.client:
            logger.error("Failed to initialize LLM client. Review functionality will be severely limited.")

    def _create_client(self) -> Optional[OpenAI]:
        """Create the OpenAI client."""
        if not self.api_key:
            logger.error("API_KEY not set or empty. LLM functionality will be limited.")
            return None

        try:
            logger.info(f"Creating OpenAI client with base URL: {self.base_url}")
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            logger.error(f"Error creating LLM client: {e}")
            return None

    async def generate_text(
        self,
        prompt: str,
    ) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: The prompt to send to the LLM
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling

        Returns:
            Generated text
        """
        if not self.client:
            error_msg = "LLM client not available. Please check API_KEY configuration."
            logger.error(error_msg)
            return error_msg

        try:
            logger.info(f"Generating text with model: {self.model}")

            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            result = response.choices[0].message.content
            logger.info(f"Successfully generated text response (length: {len(result)})")
            return result
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def generate_text_stream(
        self,
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using the LLM with streaming.

        Args:
            prompt: The prompt to send to the LLM
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling

        Yields:
            Generated text chunks
        """
        if not self.client:
            error_msg = "LLM client not available. Please check API_KEY configuration."
            logger.error(error_msg)
            yield error_msg
            return

        try:
            logger.info(f"Generating text stream with model: {self.model}")

            messages = [{"role": "user", "content": prompt}]

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error generating text stream: {e}")
            yield f"Error generating text: {str(e)}"

    async def chat(
        self,
        messages: List[Message],
    ) -> str:
        """
        Chat with the LLM.

        Args:
            messages: List of messages
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling

        Returns:
            Generated response
        """
        if not self.client:
            error_msg = "LLM client not available. Please check API_KEY configuration."
            logger.error(error_msg)
            return error_msg

        try:
            logger.info(f"Chatting with model: {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[m.to_dict() for m in messages],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error chatting: {e}")
            return f"Error chatting: {str(e)}"

    async def generate_review(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a review using the LLM.

        Args:
            prompt: The prompt to send to the LLM
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling

        Returns:
            Generated review
        """
        logger.info("Generating review")
        return await self.generate_text(prompt)
