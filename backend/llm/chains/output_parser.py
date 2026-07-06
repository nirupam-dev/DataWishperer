"""
Output parser — Extracts Python code blocks from LLM responses.

The LLM is instructed to wrap code in ```python ... ``` fences.
This parser handles edge cases: multiple code blocks, no fences,
markdown artifacts, and partial responses.
"""

from __future__ import annotations

import re
from typing import Optional

from backend.core.exceptions import GenerationError
from backend.core.logging_config import get_logger

logger = get_logger(__name__)

_CODE_FENCE_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class OutputParser:
    """
    Parses LLM output to extract executable Python code.

    Handles multiple extraction strategies in fallback order:
        1. Fenced code blocks (```python ... ```)
        2. Indented code blocks
        3. Raw output as code (last resort)
    """

    def extract_code(self, llm_output: str) -> str:
        """
        Extract Python code from the LLM's raw text output.

        Args:
            llm_output: The full text response from the LLM.

        Returns:
            The extracted Python code as a string.

        Raises:
            GenerationError: If no code could be extracted.
        """
        if not llm_output or not llm_output.strip():
            raise GenerationError("LLM returned an empty response.")

        # Strategy 1: Fenced code blocks
        code = self._extract_fenced(llm_output)
        if code:
            logger.debug("Extracted code from fenced block (%d chars)", len(code))
            return code

        # Strategy 2: Look for 'result =' assignment
        code = self._extract_by_result_assignment(llm_output)
        if code:
            logger.debug("Extracted code by result assignment (%d chars)", len(code))
            return code

        # Strategy 3: Try the entire output as code (if it looks like Python)
        code = self._extract_raw(llm_output)
        if code:
            logger.debug("Using raw output as code (%d chars)", len(code))
            return code

        raise GenerationError(
            "Could not extract Python code from the LLM response. "
            "The model may have answered in natural language instead of code."
        )

    def extract_text_response(self, llm_output: str) -> str:
        """
        Extract the natural language explanation from the LLM output.

        Strips code blocks and returns the surrounding text.

        Args:
            llm_output: The full text response from the LLM.

        Returns:
            The non-code text content.
        """
        # Remove code blocks
        text = _CODE_FENCE_PATTERN.sub("", llm_output)
        # Clean up
        text = text.strip()
        # Remove lines that are just whitespace
        lines = [line for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    # ── Private extraction strategies ────────────────────────────────────

    @staticmethod
    def _extract_fenced(text: str) -> Optional[str]:
        """Extract code from fenced markdown blocks."""
        matches = _CODE_FENCE_PATTERN.findall(text)
        if not matches:
            return None

        # If multiple blocks, concatenate them (the LLM sometimes splits code)
        code = "\n\n".join(match.strip() for match in matches)
        return code if code.strip() else None

    @staticmethod
    def _extract_by_result_assignment(text: str) -> Optional[str]:
        """Extract code by finding lines that include 'result ='."""
        lines = text.split("\n")
        code_lines: list[str] = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            # Start capturing when we see import or assignment-like code
            if (
                stripped.startswith(("import ", "from ", "df[", "df.", "result"))
                or stripped.startswith("#")
                or "=" in stripped
            ):
                in_code = True

            if in_code:
                # Stop at markdown headings or empty sections
                if stripped.startswith(("##", "**", "---")):
                    break
                code_lines.append(line)

        code = "\n".join(code_lines).strip()

        # Must contain 'result' assignment to be valid
        if code and "result" in code:
            return code
        return None

    @staticmethod
    def _extract_raw(text: str) -> Optional[str]:
        """
        Use the raw text as code if it looks like Python.

        Checks for Python-like indicators: imports, assignments, function calls.
        """
        indicators = ["import ", "df[", "df.", "result =", "result=", ".groupby(", ".mean("]
        text_lower = text.lower()

        matches = sum(1 for ind in indicators if ind in text_lower)
        if matches >= 2:
            # Strip any leading prose
            lines = text.split("\n")
            code_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(("import", "from", "df", "#", "result")):
                    code_start = i
                    break

            return "\n".join(lines[code_start:]).strip()

        return None
