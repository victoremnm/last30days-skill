"""Substack post search via Substack's public search API.

Uses substack.com/api/v1/search for post discovery.
No API key required — uses the same endpoint as the Substack web app.

Engagement: likes (subscriber reactions) and comment counts.
"""

import math
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from . import http
from .query import extract_core_subject
from .relevance import token_overlap_relevance

SUBSTACK_SEARCH_URL = "https://substack.com/api/v1/search"

DEPTH_CONFIG = {
    "quick": 10,
    "default": 25,
    "deep": 50,
}

ENRICH_LIMITS = {
    "quick": 3,
    "default": 5,
    "deep": 10,
}

# Substack post types: posts | publications
_POST_TYPE = "posts"

# Noise words to strip from Substack queries
_SUBSTACK_NOISE = frozenset({
    "best", "top", "good", "great", "awesome",
    "latest", "new", "news", "update", "updates",
    "trending", "hottest", "popular", "viral",
    "practices", "recommendations", "advice", "guide",
})


def _log(msg: str) -> None:
    """Log to stderr only in TTY mode."""
    if sys.stderr.isatty():
        sys.stderr.write(f"[Substack] {msg}\n")
        sys.stderr.flush()


def _parse_date(date_val: Any) -> Optional[str]:
    """Parse Substack post_date to YYYY-MM-DD.

    Substack returns ISO 8601 datetimes, e.g. '2024-03-15T12:00:00.000Z'.
    """
    if not date_val or not isinstance(date_val, str):
        return None
    try:
        dt = datetime.fromisoformat(date_val.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _extract_core_subject(topic: str) -> str:
    """Extract core search terms, removing Substack-specific noise."""
    return extract_core_subject(topic, noise=_SUBSTACK_NOISE)


def search_substack(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
) -> Dict[str, Any]:
    """Search Substack posts via the public search API.

    Args:
        topic: Search topic (will be simplified to core subject)
        from_date: Start date (YYYY-MM-DD) — used for post-fetch filtering
        to_date: End date (YYYY-MM-DD)
        depth: 'quick', 'default', or 'deep'

    Returns:
        Dict with 'posts' list and optional 'error' key.

    Note:
        The Substack search API doesn't support server-side date filtering,
        so we fetch more results than needed and filter client-side.
    """
    count = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["default"])
    core = _extract_core_subject(topic)

    _log(f"Searching for '{core}' (raw: '{topic}', since={from_date}, count={count})")

    params = {
        "query": core,
        "type": _POST_TYPE,
        "page": 0,
        "limit": count,
    }
    url = f"{SUBSTACK_SEARCH_URL}?{urlencode(params)}"

    try:
        response = http.request(
            "GET",
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; last30days-skill/1.0)",
            },
            timeout=30,
        )
    except http.HTTPError as e:
        _log(f"Search failed (HTTP {e.status_code}): {e}")
        return {"posts": [], "error": f"Substack search failed: {e}"}
    except Exception as e:
        _log(f"Search failed: {e}")
        return {"posts": [], "error": f"Substack search failed: {type(e).__name__}: {e}"}

    # Response may be a list (array of posts) or dict with a 'posts' key
    if isinstance(response, list):
        posts = response
    else:
        posts = response.get("posts", response.get("results", []))

    _log(f"Found {len(posts)} posts (pre-date-filter)")
    return {"posts": posts}


def parse_substack_response(
    response: Dict[str, Any],
    topic: str = "",
    from_date: str = "",
    to_date: str = "",
) -> List[Dict[str, Any]]:
    """Parse Substack API response into normalized item dicts.

    Args:
        response: Response from search_substack()
        topic: Original topic for relevance scoring
        from_date: Inclusive start date for client-side filtering (YYYY-MM-DD)
        to_date: Inclusive end date for client-side filtering (YYYY-MM-DD)

    Returns:
        List of item dicts ready for normalize_substack_items().
    """
    posts = response.get("posts", [])
    items = []

    for i, post in enumerate(posts):
        # --- Date ---
        date_str = _parse_date(post.get("post_date") or post.get("publishedAt"))

        # Client-side date filter (API has no server-side date param)
        if from_date and date_str and date_str < from_date:
            continue
        if to_date and date_str and date_str > to_date:
            continue

        # --- Identity ---
        post_id = str(post.get("id") or post.get("slug") or f"SS{i+1}")
        title = post.get("title") or ""
        subtitle = post.get("subtitle") or ""
        slug = post.get("slug") or ""

        # --- URL ---
        # Prefer canonical_url, fall back to constructing from publication subdomain
        canonical_url = post.get("canonical_url") or post.get("url") or ""
        if not canonical_url:
            pub = post.get("publication") or {}
            subdomain = pub.get("subdomain") or pub.get("custom_domain") or ""
            if subdomain and slug:
                canonical_url = f"https://{subdomain}.substack.com/p/{slug}"

        # --- Author / Publication ---
        pub = post.get("publication") or {}
        pub_name = pub.get("name") or pub.get("subdomain") or ""
        author = post.get("author") or {}
        author_name = author.get("name") or author.get("handle") or pub_name

        # --- Engagement ---
        reactions = post.get("reactions") or {}
        # Substack surfaces ❤️ reactions; fall back to reaction_count field
        likes = (
            reactions.get("❤") or
            post.get("reaction_count") or
            post.get("likes") or
            0
        )
        comment_count = post.get("comment_count") or 0

        # --- Relevance ---
        rank_score = max(0.3, 1.0 - (i * 0.02))       # positional: 1.0 → 0.3
        engagement_boost = min(0.2, math.log1p(likes + comment_count) / 30)
        if topic:
            content = f"{title} {subtitle}"
            content_score = token_overlap_relevance(topic, content)
            relevance = min(1.0, 0.55 * rank_score + 0.35 * content_score + engagement_boost)
        else:
            relevance = min(1.0, rank_score * 0.7 + engagement_boost + 0.1)

        items.append({
            "post_id": post_id,
            "title": title,
            "subtitle": subtitle,
            "url": canonical_url,
            "publication_name": pub_name,
            "author_name": author_name,
            "date": date_str,
            "engagement": {
                "likes": likes,
                "num_comments": comment_count,
            },
            "relevance": round(relevance, 2),
            "why_relevant": (
                f"Substack: {pub_name}: {title[:60]}"
                if pub_name
                else f"Substack: {title[:60]}"
            ),
        })

    return items
