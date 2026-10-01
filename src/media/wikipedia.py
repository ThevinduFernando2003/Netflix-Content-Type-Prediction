"""Poster and synopsis lookup from the public Wikipedia API.

The text and image are for display. They are never passed to the classifier.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "NetflixContentTypePrediction/1.0 (educational catalogue display)"
SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_TAG_RE = re.compile(r"<[^>]+>")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def lookup_media(title: str, content_type: str, *, timeout: float = 8.0) -> dict[str, Any]:
    """Return a Wikipedia synopsis and poster for a movie or TV title.

    ``found`` is false when the API has no usable page. Network failures stay
    in the log and also return ``found: false``.
    """
    hint = "film" if content_type == "Movie" else "television series"
    try:
        pages = _search(f"{title} {hint}", timeout=timeout)
        chosen = choose_page(pages, title, content_type)
        if chosen is None:
            pages = _search(title, timeout=timeout)
            chosen = choose_page(pages, title, content_type)
        if chosen is None:
            return _missing(title)
        summary = _summary(str(chosen["key"]), timeout=timeout)
    except httpx.HTTPError:
        logger.warning("Wikipedia lookup failed for %s", title, exc_info=True)
        return _missing(title)
    description = str(summary.get("extract") or "").strip()
    if not description or summary.get("type") == "disambiguation":
        return _missing(title)
    image = summary.get("originalimage") or summary.get("thumbnail") or {}
    if not isinstance(image, dict):
        image = {}
    content_urls = summary.get("content_urls") or {}
    urls = content_urls.get("desktop") if isinstance(content_urls, dict) else {}
    if not isinstance(urls, dict):
        urls = {}
    return {
        "found": True,
        "query": title,
        "title": str(summary.get("title") or title),
        "description": description,
        "image_url": image.get("source"),
        "page_url": urls.get("page") or f"https://en.wikipedia.org/wiki/{chosen['key']}",
        "source": "Wikipedia",
        "content_type": content_type,
    }


def choose_page(
    pages: list[dict[str, Any]], title: str, content_type: str
) -> dict[str, Any] | None:
    """Pick the search hit whose title and extract match the requested kind."""
    wanted = _normalise(title)
    best: dict[str, Any] | None = None
    best_score = 0
    for page in pages:
        page_title = str(page.get("title") or "")
        excerpt = _TAG_RE.sub("", str(page.get("excerpt") or ""))
        score = _score(page_title, excerpt, wanted, content_type)
        if score > best_score:
            best = page
            best_score = score
    return best


def _score(page_title: str, excerpt: str, wanted: str, content_type: str) -> int:
    name = _normalise(page_title)
    text = f"{page_title} {excerpt}".casefold()
    if "disambiguation" in text or "may refer to" in text or name.startswith("list of "):
        return 0
    score = 0
    if name == wanted:
        score += 6
    elif name.startswith(wanted) or wanted.startswith(name):
        score += 3
    film = any(word in text for word in ("film", "movie", "documentary"))
    television = any(word in text for word in ("television", "tv series", "miniseries", "sitcom"))
    if "series" in text and "film series" not in text:
        television = True
    if content_type == "Movie":
        score += 4 if film and not television else 2 if film else -3 if television else 0
    else:
        score += 4 if television and not film else 2 if television else -3 if film else 0
    return score


def _search(query: str, *, timeout: float) -> list[dict[str, Any]]:
    payload = _get(SEARCH_URL, timeout=timeout, params={"q": query, "limit": 5})
    pages = payload.get("pages", [])
    return pages if isinstance(pages, list) else []


def _summary(key: str, *, timeout: float) -> dict[str, Any]:
    body = _get(SUMMARY_URL.format(title=key), timeout=timeout)
    return body if isinstance(body, dict) else {}


def _get(url: str, *, timeout: float, params: dict[str, Any] | None = None) -> dict[str, Any]:
    with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        body = response.json()
    return body if isinstance(body, dict) else {}


def _normalise(text: str) -> str:
    return _NON_WORD_RE.sub(" ", text.casefold().replace("_", " ")).strip()


def _missing(title: str) -> dict[str, Any]:
    return {
        "found": False,
        "query": title,
        "title": title,
        "description": None,
        "image_url": None,
        "page_url": None,
        "source": "Wikipedia",
        "content_type": None,
    }
