"""Robust web scraper with retries, realistic User-Agent, HTML cleaning, and text chunking."""

import re
import time
import logging
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from src.schemas import ScrapedURLData

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def clean_html(html_content: str) -> tuple[str, str]:
    """Clean raw HTML: remove scripts/navs/styles, extract title and normalized readable text."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Remove non-content tags
    unwanted_tags = [
        "script", "style", "noscript", "nav", "footer", "header",
        "aside", "svg", "form", "button", "iframe", "menu", "ad"
    ]
    for tag in soup(unwanted_tags):
        tag.decompose()

    # Prioritize main article container if available
    main_container = soup.find("article") or soup.find("main") or soup.find(id=re.compile(r"content|body|main", re.I))
    target = main_container if main_container else soup.body or soup

    # Extract text from paragraphs, headers, and list items
    content_blocks = []
    for element in target.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        text = element.get_text(separator=" ", strip=True)
        if text and len(text) > 20:  # Skip trivial labels
            content_blocks.append(text)

    # Fallback to general get_text if block extraction was too sparse
    if not content_blocks:
        raw_text = target.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in raw_text.splitlines() if len(line.strip()) > 30]
        full_text = "\n\n".join(lines)
    else:
        full_text = "\n\n".join(content_blocks)

    # Clean multiple spaces and normalize newlines
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    return title, full_text


def chunk_text(text: str, target_chunk_size: int = 1200, min_chunk_size: int = 300) -> List[str]:
    """Split cleaned text into cohesive chunks respecting paragraph or sentence boundaries."""
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for p in paragraphs:
        p_len = len(p)
        if current_length + p_len > target_chunk_size and current_length >= min_chunk_size:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_length = p_len
        else:
            current_chunk.append(p)
            current_length += p_len + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # If text has no paragraph breaks, split by sentence
    if len(chunks) <= 1 and len(text) > target_chunk_size:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_length = 0
        for s in sentences:
            s_len = len(s)
            if current_length + s_len > target_chunk_size and current_length >= min_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [s]
                current_length = s_len
            else:
                current_chunk.append(s)
                current_length += s_len + 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))

    return chunks


class WebScraper:
    """Production web scraper with retries, timeout, realistic headers, and error handling."""

    def __init__(
        self,
        timeout: int = 15,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        headers: Optional[dict] = None
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.headers = headers or DEFAULT_HEADERS
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def scrape_url(self, url: str, index: int = 1) -> ScrapedURLData:
        """Scrape a single URL with retries, clean text, and chunk it."""
        url_clean = url.strip()
        last_error = None
        status_code = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"[{index}] Fetching URL (attempt {attempt}/{self.max_retries}): {url_clean}")
                response = self.session.get(url_clean, timeout=self.timeout, allow_redirects=True)
                status_code = response.status_code

                if response.status_code == 200:
                    title, text = clean_html(response.text)
                    chunks = chunk_text(text)
                    logger.info(f"[{index}] Successfully scraped {len(text)} chars ({len(chunks)} chunks).")
                    return ScrapedURLData(
                        url=url_clean,
                        index=index,
                        title=title,
                        status_code=status_code,
                        success=True,
                        cleaned_text=text,
                        chunks=chunks
                    )
                else:
                    last_error = f"HTTP {response.status_code}: {response.reason}"
                    logger.warning(f"[{index}] Received {last_error} on attempt {attempt}")
                    if response.status_code in [403, 401, 404]:
                        # Do not retry on permanent client errors/blocks
                        break

            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
                logger.warning(f"[{index}] Error on attempt {attempt}: {last_error}")

            if attempt < self.max_retries:
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_time)

        logger.error(f"[{index}] Failed to scrape {url_clean}: {last_error}")
        return ScrapedURLData(
            url=url_clean,
            index=index,
            status_code=status_code,
            success=False,
            error_message=last_error or "Unknown scraping failure",
            cleaned_text="",
            chunks=[]
        )
