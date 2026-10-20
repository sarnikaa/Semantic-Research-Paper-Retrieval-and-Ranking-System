"""
Text highlighting utilities for search engine snippet display.
Highlights query keywords in paper abstracts using HTML <mark> styling.
"""

import re
from typing import Set

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours"
}


def highlight_query_terms(text: str, query: str) -> str:
    """
    Highlights non-stopword query keywords inside the text with warm yellow markup.

    Args:
        text: Target paper abstract or title
        query: User search query

    Returns:
        HTML string with <mark> tags around matching terms
    """
    if not query or not text:
        return text

    # Extract query tokens (length >= 3, excluding stopwords)
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query.lower())
    valid_tokens = [re.escape(t) for t in raw_tokens if t not in STOPWORDS]

    if not valid_tokens:
        return text

    pattern = re.compile(rf"\b({'|'.join(valid_tokens)})\b", re.IGNORECASE)
    
    highlighted = pattern.sub(
        r'<mark style="background-color: #fff3bf; color: #1a1a1a; padding: 1px 4px; border-radius: 3px; font-weight: 600;">\g<0></mark>',
        text
    )
    return highlighted
