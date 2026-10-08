"""Walking the builder's cursor listings without trusting the server's shape.

Split out of ``builder_api`` because these are the rules for *following* a
listing, not for speaking to any one endpoint: the client contributes the URL
and the response key, and everything about when to stop lives here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Final

import requests

# The builder clamps a page to 100 rows, so this ceiling is tens of thousands of
# builds — orders of magnitude past any real workspace. It bounds a *server*
# that keeps handing back a cursor, not a listing anyone actually has.
_MAX_LIST_PAGES: Final = 200
# Sent explicitly: httpkit.ClampLimit falls back to 20 rows, not to its 100-row
# ceiling, when the caller names no limit — so an unset limit pages at a fifth of
# what `_MAX_LIST_PAGES` above was sized against, and fails at 4,000 rows.
PAGE_LIMIT: Final = 100
# `list_blobs` is served as a single page with no cursor, so a clamped listing is
# indistinguishable from a complete one except by its length: the builder returns
# httpkit.DefaultPageLimit rows when the caller names no limit, and this client
# names none. A page of exactly that many rows is the ceiling, not the whole set.
_BLOB_PAGE_LIMIT: Final = 20


class BuilderPaginationError(requests.RequestException):
    """A cursor listing cannot be walked to its end without trusting the server.

    Subclasses ``RequestException`` rather than ``ValueError`` so that
    ``build._builder_call`` reports it as ``build_builder_error`` ("builder call
    failed"), which is what it is. That wrapper's ``ValueError`` clause is
    reserved for *caller* input mistakes and would relabel this
    ``build_missing_input``, sending the user off to fix a flag that is fine.
    """


def cursor_pages(fetch: Callable[[str | None], dict], listing: str) -> Iterator[dict]:
    """Yield every page of a builder cursor listing, or refuse to keep asking.

    A cursor is followed only when the server returns it as a non-empty **str**.
    ``nextCursor: 5`` or ``{"a": 1}`` is truthy and survives ``urlencode``, so an
    untyped walk would page forever against a server whose shape drifted, with
    the accumulated rows growing without bound — the failure this listing's
    ``deploy_api.iter_deployments`` sibling already guards against.

    A repeated cursor and an exceeded page cap raise rather than returning what
    was collected so far: a short listing presented as complete is the silent
    truncation this pagination was added to fix.
    """
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(_MAX_LIST_PAGES):
        page = fetch(cursor)
        yield page
        next_cursor = page.get("nextCursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            return
        if next_cursor in seen:
            raise BuilderPaginationError(f"builder repeated a {listing} page cursor; refusing to page in a circle")
        seen.add(next_cursor)
        cursor = next_cursor
    raise BuilderPaginationError(f"builder {listing} listing did not end within {_MAX_LIST_PAGES} pages")


def blob_listing_is_clamped(blobs: list[dict]) -> bool:
    """Whether ``BuilderClient.list_blobs`` returned a full page, i.e. one the
    server may have cut short."""
    return len(blobs) >= _BLOB_PAGE_LIMIT
