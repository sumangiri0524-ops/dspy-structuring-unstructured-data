"""Unit tests for web scraper, HTML cleaning, and text chunking."""

from unittest.mock import patch, MagicMock
import requests
from src.scraper import clean_html, chunk_text, WebScraper


def test_clean_html_removes_scripts_and_nav():
    sample_html = """
    <html>
      <head><title>Sustainable Farming Systems</title><script>alert('ad');</script></head>
      <body>
        <nav><a href='/home'>Home</a></nav>
        <header>Header Bar</header>
        <main>
          <p>Sustainable agriculture improves soil health and water conservation across regions.</p>
          <p>Nitrogen uptake is enhanced through diverse crop rotation techniques.</p>
        </main>
        <footer>Footer info</footer>
      </body>
    </html>
    """
    title, text = clean_html(sample_html)
    assert title == "Sustainable Farming Systems"
    assert "alert" not in text
    assert "Sustainable agriculture improves soil health" in text
    assert "Nitrogen uptake is enhanced" in text


def test_chunk_text_splits_coherent_blocks():
    long_text = ("Sustainable agriculture represents farming in sustainable ways. " * 30)
    chunks = chunk_text(long_text, target_chunk_size=500, min_chunk_size=200)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) > 100


def test_scraper_success_mock():
    scraper = WebScraper(timeout=5, max_retries=1)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><head><title>Test Page</title></head><body><p>Valid content paragraph for testing.</p></body></html>"

    with patch.object(scraper.session, "get", return_value=mock_resp):
        res = scraper.scrape_url("https://example.com/valid", index=1)
        assert res.success is True
        assert res.status_code == 200
        assert res.title == "Test Page"
        assert "Valid content paragraph" in res.cleaned_text


def test_scraper_handles_http_403_gracefully():
    scraper = WebScraper(timeout=5, max_retries=1)
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.reason = "Forbidden"

    with patch.object(scraper.session, "get", return_value=mock_resp):
        res = scraper.scrape_url("https://example.com/blocked", index=2)
        assert res.success is False
        assert res.status_code == 403
        assert "403" in (res.error_message or "")
        assert res.cleaned_text == ""  # Never fabricate info on failure
