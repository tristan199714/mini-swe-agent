import re


def extract_score(llm_output: str) -> int:
    """Extract score from LLM output.

    Accepted labels include:
      - #thescore: <n>
      - thescore: <n>
      - score: <n>
      - final score is <n>
      - {"score": <n>}
    Valid scores: 1-10 (continuous scale)
    """
    if not llm_output:
        return -1

    text = llm_output.strip()
    # Normalize common full-width punctuation/digits
    trans = str.maketrans("０１２３４５６７８９：", "0123456789:")
    text = text.translate(trans)

    patterns = [
        r"#\s*thescore\s*:\s*(\d{1,2})\b",
        r"\bthescore\s*:\s*(\d{1,2})\b",
        r"\b(?:final|overall)?\s*score(?:\s+is)?\s*[:=]?\s*(\d{1,2})\b",
        r"\bscore\s+of\s+(\d{1,2})\b",
        r"\bscore\s+would\s+be\s+(\d{1,2})\b",
        r"\b(?:assign|assigned|give|given|rate|rated)\b[^\n]{0,30}\bscore\b[^\n]{0,20}\b(\d{1,2})\b",
        r"\b(\d{1,2})\s*/\s*10\b",
        r'(?:评分|得分|分数)\s*(?:为|是|:|：)?\s*(\d{1,2})\b',
        r'"score"\s*:\s*(\d{1,2})\b',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 10:
                return val

    # Fallback: last line might be just the number
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.isdigit() and 1 <= int(s) <= 10:
            return int(s)

    return -1
