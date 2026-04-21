"""Fetch recent arXiv papers for a company (cs.AI, cs.LG, cs.CL). No API key required."""
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# arXiv API namespace for parsing XML (their response uses it)
ARXiv_NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_recent_papers(company: str, days: int = 60):
    """
    Fetch up to 8 recent papers from arXiv for the company, in cs.AI, cs.LG, cs.CL.
    Returns a list of dicts: title, summary (first 300 chars), published (date, first 10 chars).
    """
    # Build query: company name and (cat:cs.AI OR cat:cs.LG OR cat:cs.CL), sorted by submittedDate
    query = f'all:"{company}" AND (cat:cs.AI OR cat:cs.LG OR cat:cs.CL)'
    params = urllib.parse.urlencode(
        {"search_query": query, "sortBy": "submittedDate", "sortOrder": "descending", "max_results": 8}
    )
    url = "https://export.arxiv.org/api/query?" + params
    with urllib.request.urlopen(url, timeout=15) as resp:
        raw = resp.read()
    # Parse XML; ElementTree parses the response into a tree of elements
    root = ET.fromstring(raw)
    out = []
    # Iterate over each <entry> (one per paper); find them using the atom namespace
    for entry in root.findall("atom:entry", ARXiv_NS):
        # find() returns the first matching child element; namespace required for atom tags
        title_el = entry.find("atom:title", ARXiv_NS)
        title = (title_el.text or "").strip().replace("\n", " ")
        summary_el = entry.find("atom:summary", ARXiv_NS)
        summary = (summary_el.text or "")[:300].strip().replace("\n", " ")
        published_el = entry.find("atom:published", ARXiv_NS)
        published = (published_el.text or "")[:10]
        # List comprehension could build this; we build one dict per paper for clarity
        out.append({"title": title, "summary": summary, "published": published})
    return out
