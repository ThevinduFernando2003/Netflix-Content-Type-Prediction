"""Wikipedia poster lookup. Tests never call the network."""

from __future__ import annotations

from fastapi.testclient import TestClient
from src.media.wikipedia import choose_page, lookup_media


def test_choose_page_prefers_the_exact_series_over_a_spinoff() -> None:
    pages = [
        {
            "key": "Stranger_Things:_Tales_from_'85",
            "title": "Stranger Things: Tales from '85",
            "excerpt": "An animated television series and a spin-off.",
        },
        {
            "key": "Stranger_Things",
            "title": "Stranger Things",
            "excerpt": "An American science fiction television series.",
        },
    ]
    chosen = choose_page(pages, "Stranger Things", "TV Show")
    assert chosen is not None
    assert chosen["key"] == "Stranger_Things"


def test_choose_page_rejects_a_bare_disambiguation() -> None:
    pages = [
        {"key": "Midnight_Mass", "title": "Midnight Mass", "excerpt": "Midnight Mass may refer to:"}
    ]
    assert choose_page(pages, "Midnight Mass", "TV Show") is None


def test_lookup_media_uses_the_summary_poster(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.media.wikipedia._search",
        lambda query, timeout: [
            {"key": "Inception", "title": "Inception", "excerpt": "A 2010 science fiction film."}
        ],
    )
    monkeypatch.setattr(
        "src.media.wikipedia._summary",
        lambda key, timeout: {
            "type": "standard",
            "title": "Inception",
            "extract": "Inception is a 2010 science fiction film.",
            "originalimage": {"source": "https://example.test/inception.jpg"},
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Inception"}},
        },
    )
    media = lookup_media("Inception", "Movie")
    assert media["found"] is True
    assert media["image_url"] == "https://example.test/inception.jpg"
    assert media["description"].startswith("Inception is a 2010")
    assert media["source"] == "Wikipedia"


def test_media_endpoint_returns_the_lookup(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.api.main.lookup_media",
        lambda title, content_type: {
            "found": True,
            "query": title,
            "title": title,
            "description": "A film about a dream.",
            "image_url": "https://example.test/poster.jpg",
            "page_url": "https://en.wikipedia.org/wiki/Inception",
            "source": "Wikipedia",
            "content_type": content_type,
        },
    )
    response = client.get("/media", params={"title": "Inception", "content_type": "Movie"})
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["image_url"] == "https://example.test/poster.jpg"
    assert body["description"] == "A film about a dream."


def test_media_endpoint_rejects_a_blank_title(client: TestClient) -> None:
    response = client.get("/media", params={"title": ""})
    assert response.status_code == 422
