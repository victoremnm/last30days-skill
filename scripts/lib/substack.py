"""Substack post retrieval via per-publication and Brave search.

Two retrieval modes:
  1. Feed mode (SUBSTACK_FEEDS env var): fetch recent posts from a
     user-configured list of Substack publication URLs.  Uses each
     publication's /api/v1/posts endpoint — no auth required, rich
     engagement data (reaction_count, comment_count, restacks).

  2. Discovery mode (BRAVE_API_KEY or fallback): use Brave search
     filtered to site:substack.com to discover cross-publication posts
     for a given topic, then enrich each result with publication API
     metadata where possible.

The two modes are merged and deduplicated by canonical_url before
being returned to the pipeline.
"""

import math
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

from . import http
from .query import extract_core_subject
from .relevance import token_overlap_relevance

# ── Per-publication API ──────────────────────────────────────────────────────

PUB_POSTS_PATH = "/api/v1/posts"
PUB_POSTS_PARAMS = "?limit={limit}&type=newsletter"

DEPTH_CONFIG = {
    "quick": 15,
    "default": 30,
    "deep": 60,
}

# ── Brave discovery API ──────────────────────────────────────────────────────

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# ── Noise words to strip from Substack queries ───────────────────────────────

_SUBSTACK_NOISE = frozenset({
    "best", "top", "good", "great", "awesome",
    "latest", "new", "news", "update", "updates",
    "trending", "hottest", "popular", "viral",
    "practices", "recommendations", "advice", "guide",
})


def _log(msg: str) -> None:
    if sys.stderr.isatty():
        sys.stderr.write(f"[Substack] {msg}\n")
        sys.stderr.flush()


def _parse_date(date_val: Any) -> Optional[str]:
    """Parse ISO 8601 datetime to YYYY-MM-DD."""
    if not date_val or not isinstance(date_val, str):
        return None
    try:
        dt = datetime.fromisoformat(date_val.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _extract_core_subject(topic: str) -> str:
    return extract_core_subject(topic, noise=_SUBSTACK_NOISE)


def _pub_base_url(url: str) -> str:
    """Normalise a publication URL to its base (scheme + netloc)."""
    parsed = urlparse(url.strip().rstrip("/"))
    if not parsed.scheme:
        parsed = urlparse("https://" + url.strip().rstrip("/"))
    return f"{parsed.scheme}://{parsed.netloc}"


def _fetch_publication_posts(
    base_url: str,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """Fetch recent posts from a single Substack publication.

    Args:
        base_url: Publication root URL, e.g. 'https://www.ignorance.ai'
        limit: Max posts to fetch

    Returns:
        List of raw post dicts from the Substack API.
    """
    url = f"{base_url}{PUB_POSTS_PATH}?limit={limit}&type=newsletter"
    try:
        posts = http.request(
            "GET",
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; last30days-skill/1.0)",
            },
            timeout=20,
        )
        if isinstance(posts, list):
            return posts
        # Some publications wrap in {"posts": [...]}
        if isinstance(posts, dict):
            return posts.get("posts", [])
        return []
    except Exception as e:
        _log(f"Failed to fetch posts from {base_url}: {e}")
        return []


def _enrich_with_pub_api(canonical_url: str) -> Optional[Dict[str, Any]]:
    """Try to fetch post metadata from the publication API by post URL.

    Given a canonical URL like https://blog.dataexpert.io/p/parquet-format,
    hits /api/v1/posts/{slug} for richer metadata.
    """
    parsed = urlparse(canonical_url)
    # Extract slug from /p/{slug} paths
    parts = parsed.path.split("/")
    if len(parts) < 3 or parts[1] != "p":
        return None
    slug = parts[2]
    base = f"{parsed.scheme}://{parsed.netloc}"
    url = f"{base}/api/v1/posts/{slug}"
    try:
        data = http.request(
            "GET", url,
            headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if isinstance(data, dict) and data.get("title"):
            return data
    except Exception:
        pass
    return None


# ── Mode 1: Feed-based retrieval ─────────────────────────────────────────────

def fetch_from_feeds(
    feeds: List[str],
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
) -> List[Dict[str, Any]]:
    """Fetch and keyword-filter posts from configured Substack publication URLs.

    Args:
        feeds: List of publication base URLs
        topic: Research topic for relevance filtering
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        depth: 'quick', 'default', or 'deep'

    Returns:
        List of raw item dicts.
    """
    limit = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["default"])
    per_feed = max(10, limit // max(len(feeds), 1))

    items = []
    seen_urls: set = set()

    for feed_url in feeds:
        base = _pub_base_url(feed_url)
        _log(f"Fetching from {base} (limit={per_feed})")
        posts = _fetch_publication_posts(base, limit=per_feed)

        for post in posts:
            canonical_url = post.get("canonical_url", "")
            if canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)

            date_str = _parse_date(post.get("post_date"))

            # Date filter
            if from_date and date_str and date_str < from_date:
                continue
            if to_date and date_str and date_str > to_date:
                continue

            item = _build_item_from_pub_post(post, i=len(items), topic=topic)
            if item:
                items.append(item)

    _log(f"Feed mode: {len(items)} posts from {len(feeds)} feed(s)")
    return items


# ── Mode 2: Brave-search discovery ───────────────────────────────────────────

def discover_via_brave(
    topic: str,
    from_date: str,
    to_date: str,
    brave_api_key: str,
    depth: str = "default",
) -> List[Dict[str, Any]]:
    """Discover Substack posts via Brave search filtered to site:substack.com.

    Args:
        topic: Research topic
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        brave_api_key: Brave Search API key
        depth: 'quick', 'default', or 'deep'

    Returns:
        List of raw item dicts (possibly enriched from publication API).
    """
    count = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["default"])
    core = _extract_core_subject(topic)
    query = f"site:substack.com {core}"

    _log(f"Brave discovery: '{query}' (count={count})")

    params = {
        "q": query,
        "count": str(min(count, 20)),
        "result_filter": "web",
        "freshness": "pm",  # past month
    }
    url = f"{BRAVE_SEARCH_URL}?{urlencode(params)}"

    try:
        data = http.request(
            "GET",
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": brave_api_key,
            },
            timeout=20,
        )
    except Exception as e:
        _log(f"Brave discovery failed: {e}")
        return []

    results = data.get("web", {}).get("results", [])
    _log(f"Brave returned {len(results)} results")

    items = []
    seen_urls: set = set()

    for i, result in enumerate(results):
        url_str = result.get("url", "")
        if not url_str or url_str in seen_urls:
            continue
        # Only keep post URLs (contain /p/ in path)
        if "/p/" not in urlparse(url_str).path:
            continue
        seen_urls.add(url_str)

        # Try to enrich with publication API
        pub_post = _enrich_with_pub_api(url_str)
        if pub_post:
            item = _build_item_from_pub_post(pub_post, i=i, topic=topic)
        else:
            item = _build_item_from_brave_result(result, i=i, topic=topic, from_date=from_date, to_date=to_date)

        if item:
            # Apply date filter
            date_str = item.get("date")
            if from_date and date_str and date_str < from_date:
                continue
            if to_date and date_str and date_str > to_date:
                continue
            items.append(item)

    return items


# ── Shared builder helpers ────────────────────────────────────────────────────

def _build_item_from_pub_post(
    post: Dict[str, Any],
    i: int,
    topic: str,
) -> Optional[Dict[str, Any]]:
    """Build a normalizable item dict from a Substack publication API post."""
    title = post.get("title") or ""
    if not title:
        return None

    subtitle = post.get("subtitle") or ""
    canonical_url = post.get("canonical_url") or ""
    date_str = _parse_date(post.get("post_date"))

    # Engagement — publication API uses reaction_count / comment_count / restacks
    reaction_count = post.get("reaction_count") or 0
    # reactions dict may have ❤ key
    reactions = post.get("reactions") or {}
    likes = reactions.get("❤") or reaction_count
    comment_count = post.get("comment_count") or 0
    restacks = post.get("restacks") or 0

    # Author from publishedBylines
    bylines = post.get("publishedBylines") or []
    author_name = bylines[0].get("name", "") if bylines else ""
    # Publication name from byline publication info
    pub_name = ""
    if bylines:
        pub_info = (bylines[0].get("publicationUsers") or [{}])[0].get("publication") or {}
        pub_name = pub_info.get("name") or ""

    # Relevance
    rank_score = max(0.3, 1.0 - (i * 0.02))
    total_engagement = likes + comment_count + restacks
    engagement_boost = min(0.2, math.log1p(total_engagement) / 30)
    content = f"{title} {subtitle} {post.get('truncated_body_text', '')[:300]}"
    content_score = token_overlap_relevance(topic, content) if topic else 0.5
    relevance = min(1.0, 0.50 * rank_score + 0.40 * content_score + engagement_boost)

    return {
        "post_id": str(post.get("id") or post.get("slug") or f"SS{i+1}"),
        "title": title,
        "subtitle": subtitle,
        "url": canonical_url,
        "publication_name": pub_name,
        "author_name": author_name,
        "date": date_str,
        "engagement": {
            "likes": likes,
            "num_comments": comment_count,
            "restacks": restacks,
        },
        "relevance": round(relevance, 2),
        "why_relevant": (
            f"Substack: {pub_name}: {title[:60]}"
            if pub_name
            else f"Substack: {title[:60]}"
        ),
    }


def _build_item_from_brave_result(
    result: Dict[str, Any],
    i: int,
    topic: str,
    from_date: str = "",
    to_date: str = "",
) -> Optional[Dict[str, Any]]:
    """Build a normalizable item from a Brave search result (no pub API data)."""
    title = result.get("title") or ""
    url_str = result.get("url") or ""
    if not title or not url_str:
        return None

    description = result.get("description") or ""
    date_str = _parse_date(result.get("page_age") or result.get("age"))

    # Derive publication name from URL
    parsed = urlparse(url_str)
    hostname = parsed.netloc
    # Strip www. prefix
    pub_name = hostname.replace("www.", "")

    rank_score = max(0.3, 1.0 - (i * 0.02))
    content_score = token_overlap_relevance(topic, f"{title} {description}") if topic else 0.5
    relevance = min(1.0, 0.55 * rank_score + 0.45 * content_score)

    return {
        "post_id": f"SS_B{i+1}",
        "title": title,
        "subtitle": description[:200],
        "url": url_str,
        "publication_name": pub_name,
        "author_name": "",
        "date": date_str,
        "engagement": {"likes": 0, "num_comments": 0, "restacks": 0},
        "relevance": round(relevance, 2),
        "why_relevant": f"Substack (via Brave): {pub_name}: {title[:60]}",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def search_substack(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search Substack posts via feeds and/or Brave discovery.

    Mode selection:
    - If SUBSTACK_FEEDS is configured, fetch from those publications.
    - If BRAVE_API_KEY is configured, also run Brave site:substack.com discovery.
    - If neither, return an error (no zero-config discovery available).

    Args:
        topic: Research topic
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        depth: 'quick', 'default', or 'deep'
        config: Config dict (may contain SUBSTACK_FEEDS, BRAVE_API_KEY)

    Returns:
        Dict with 'posts' list and optional 'error'.
    """
    config = config or {}

    # Parse configured feeds
    feeds_raw = config.get("SUBSTACK_FEEDS", "")
    feeds = [f.strip() for f in feeds_raw.split(",") if f.strip()] if feeds_raw else []

    brave_key = config.get("BRAVE_API_KEY", "")

    if not feeds and not brave_key:
        return {
            "posts": [],
            "error": (
                "No Substack source configured. "
                "Set SUBSTACK_FEEDS=https://newsletter1.substack.com,https://pub2.com "
                "and/or BRAVE_API_KEY for cross-publication discovery."
            ),
        }

    all_posts: List[Dict[str, Any]] = []
    seen_urls: set = set()

    # Mode 1: configured feeds
    if feeds:
        feed_items = fetch_from_feeds(feeds, topic, from_date, to_date, depth)
        for item in feed_items:
            url = item.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                all_posts.append(item)

    # Mode 2: Brave discovery
    if brave_key:
        brave_items = discover_via_brave(topic, from_date, to_date, brave_key, depth)
        for item in brave_items:
            url = item.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                all_posts.append(item)

    _log(f"Total posts (before pipeline): {len(all_posts)}")
    return {"posts": all_posts}


def parse_substack_response(
    response: Dict[str, Any],
    topic: str = "",
    from_date: str = "",
    to_date: str = "",
) -> List[Dict[str, Any]]:
    """Return posts from the search_substack() response.

    Items are already built as normalizable dicts by search_substack(),
    so this is a passthrough with an optional final date safety check.
    """
    posts = response.get("posts", [])
    if not (from_date or to_date):
        return posts

    # Final date safety net
    result = []
    for item in posts:
        date_str = item.get("date")
        if from_date and date_str and date_str < from_date:
            continue
        if to_date and date_str and date_str > to_date:
            continue
        result.append(item)

    return result
