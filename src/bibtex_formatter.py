"""
Multi-Format Academic Citation Formatter.
Generates standardized citations in:
1. BibTeX
2. APA 7th Edition
3. IEEE Format
4. RIS (Research Information Systems format for Zotero, Mendeley, EndNote)
"""

import re
from typing import Dict, Any


def get_authors_list(doc: Dict[str, Any]) -> list:
    """Helper to extract author list consistently."""
    authors = doc.get("authors", "")
    if isinstance(authors, list):
        return [str(a).strip() for a in authors if str(a).strip()]
    elif isinstance(authors, str) and authors.strip():
        # Split by comma or 'and'
        raw = re.split(r",|\sand\s", authors)
        return [r.strip() for r in raw if r.strip()]
    return ["Anonymous"]


def format_bibtex(doc_id: str, doc: Dict[str, Any]) -> str:
    """Generates standardized BibTeX entry."""
    title = doc.get("title", "Untitled Document").strip()
    year = str(doc.get("year", "2024"))
    authors_list = get_authors_list(doc)
    author_str = " and ".join(authors_list)

    clean_id = re.sub(r"\W+", "_", str(doc_id))
    citation_key = f"paper_{clean_id}"

    return (
        f"@article{{{citation_key},\n"
        f"  title = {{{{{title}}}}},\n"
        f"  author = {{{author_str}}},\n"
        f"  year = {{{year}}},\n"
        f"  journal = {{arXiv preprint / SciFact Archive}},\n"
        f"  url = {{https://arxiv.org/abs/{doc_id}}},\n"
        f"  note = {{Document ID: {doc_id}}}\n"
        f"}}"
    )


def to_apa_author(name: str) -> str:
    """Formats 'First Last' to APA style 'Last, F.'"""
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {parts[0][0]}."
    return name


def format_apa7(doc_id: str, doc: Dict[str, Any]) -> str:
    """Generates APA 7th Edition citation."""
    title = doc.get("title", "Untitled Document").strip()
    year = str(doc.get("year", "2024"))
    raw_authors = get_authors_list(doc)
    authors = [to_apa_author(a) for a in raw_authors]

    if len(authors) == 1:
        auth_str = authors[0]
    elif len(authors) == 2:
        auth_str = f"{authors[0]}, & {authors[1]}"
    elif len(authors) > 2:
        auth_str = f"{authors[0]}, et al."
    else:
        auth_str = "Anonymous"

    return f"{auth_str} ({year}). {title}. arXiv preprint arXiv:{doc_id}."


def format_ieee(doc_id: str, doc: Dict[str, Any]) -> str:
    """Generates IEEE style citation."""
    title = doc.get("title", "Untitled Document").strip()
    year = str(doc.get("year", "2024"))
    authors = get_authors_list(doc)

    if len(authors) == 1:
        auth_str = authors[0]
    elif len(authors) <= 3:
        auth_str = ", ".join(authors[:-1]) + f", and {authors[-1]}"
    else:
        auth_str = f"{authors[0]} et al."

    return f"{auth_str}, \"{title},\" arXiv:{doc_id}, {year}."


def format_ris(doc_id: str, doc: Dict[str, Any]) -> str:
    """Generates RIS file format for Zotero / Mendeley / EndNote."""
    title = doc.get("title", "Untitled Document").strip()
    year = str(doc.get("year", "2024"))
    authors = get_authors_list(doc)

    lines = [
        "TY  - JOUR",
        f"TI  - {title}",
        f"PY  - {year}",
        f"ID  - {doc_id}",
        f"UR  - https://arxiv.org/abs/{doc_id}",
        "JO  - arXiv Preprint / Scientific Repository"
    ]
    for author in authors:
        lines.append(f"AU  - {author}")
    lines.append("ER  - \n")
    return "\n".join(lines)


def format_all_citations(doc_id: str, doc: Dict[str, Any]) -> Dict[str, str]:
    """Returns a dictionary containing all citation formats."""
    return {
        "BibTeX": format_bibtex(doc_id, doc),
        "APA 7th": format_apa7(doc_id, doc),
        "IEEE": format_ieee(doc_id, doc),
        "RIS": format_ris(doc_id, doc)
    }
