"""PR Cache Management for temporary PR testing.

This module provides functionality for caching built frontend PRs to enable
quick switching between different PR versions without rebuilding.
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from rich import print as rprint

from comfy_cli.config_manager import ConfigManager


class PRCache:
    """Manages cached PR builds for quick switching.

    This class handles the caching of built frontend PRs, including:
    - Cache directory management
    - Cache validity checking with age limits
    - Automatic cleanup of old/excess cache entries
    - Human-readable cache information display
    """

    # Default cache settings
    DEFAULT_MAX_CACHE_AGE_DAYS = 7  # Cache entries older than this are considered stale
    DEFAULT_MAX_CACHE_ITEMS = 10  # Maximum number of cached PRs to keep

    def __init__(self) -> None:
        """Initialize PR cache with default settings."""
        self.cache_dir = Path(ConfigManager().get_config_path()) / "pr-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_age = timedelta(days=self.DEFAULT_MAX_CACHE_AGE_DAYS)
        self.max_cache_items = self.DEFAULT_MAX_CACHE_ITEMS

    def get_frontend_cache_path(self, pr_info) -> Path:
        """Get cache path for a frontend PR"""
        # Use PR number and repo as cache key
        cache_key = f"{pr_info.user}-{pr_info.number}-{pr_info.head_branch}"
        # Sanitize for filesystem
        cache_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in cache_key)
        return self.cache_dir / "frontend" / cache_key

    def get_cache_info_path(self, cache_path: Path) -> Path:
        """Get path to cache info file"""
        return cache_path / ".cache-info.json"

    def is_cache_valid(self, pr_info, cache_path: Path) -> bool:
        """Check if cached build is still valid"""
        info_path = self.get_cache_info_path(cache_path)
        if not info_path.exists():
            return False

        try:
            with open(info_path, encoding="utf-8") as file:
                cache_info = json.load(file)

            # Check if cache metadata matches
            if not (
                cache_info.get("pr_number") == pr_info.number
                and cache_info.get("head_branch") == pr_info.head_branch
                and cache_info.get("user") == pr_info.user
            ):
                return False

            # Check if cache is too old
            cached_at = cache_info.get("cached_at")
            if cached_at:
                cache_time = datetime.fromisoformat(cached_at)
                if datetime.now() - cache_time > self.max_cache_age:
                    return False

            return True
        except (json.JSONDecodeError, OSError):
            return False

    def save_cache_info(self, pr_info, cache_path: Path) -> None:
        """Save cache metadata."""
        info_path = self.get_cache_info_path(cache_path)
        cache_info = {
            "pr_number": pr_info.number,
            "pr_title": pr_info.title,
            "user": pr_info.user,
            "head_branch": pr_info.head_branch,
            "head_repo_url": pr_info.head_repo_url,
            "cached_at": datetime.now().isoformat(),
        }

        with open(info_path, "w", encoding="utf-8") as file:
            json.dump(cache_info, file, indent=2)

        # Enforce cache limits after saving new cache
        self.enforce_cache_limits()

    def get_cached_frontend_path(self, pr_info) -> Optional[Path]:
        """Get path to cached frontend build if valid"""
        cache_path = self.get_frontend_cache_path(pr_info)
        dist_path = cache_path / "repo" / "dist"

        if dist_path.exists() and self.is_cache_valid(pr_info, cache_path):
            return dist_path
        return None

    def _load_cache_info(self, cache_dir: Path) -> Optional[dict]:
        """Load cache info from a directory."""
        info_path = self.get_cache_info_path(cache_dir)
        if not info_path.exists():
            return None

        try:
            with open(info_path, encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return None

    def _clean_specific_pr_cache(self, frontend_cache: Path, pr_number: int) -> None:
        """Clean cache for a specific PR number."""
        for cache_dir in frontend_cache.iterdir():
            if not cache_dir.is_dir():
                continue
            info = self._load_cache_info(cache_dir)
            if info and info.get("pr_number") == pr_number:
                rprint(f"[yellow]Removing cache for PR #{pr_number}[/yellow]")
                shutil.rmtree(cache_dir)
                break

    def clean_frontend_cache(self, pr_number: Optional[int] = None) -> None:
        """Clean frontend cache (specific PR or all)."""
        frontend_cache = self.cache_dir / "frontend"
        if not frontend_cache.exists():
            return

        if pr_number:
            self._clean_specific_pr_cache(frontend_cache, pr_number)
        else:
            # Clean all
            rprint("[yellow]Removing all frontend PR cache[/yellow]")
            shutil.rmtree(frontend_cache)

    def _calculate_cache_size_mb(self, cache_dir: Path) -> float:
        """Calculate the size of a cache directory in MB."""
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        return total_size / (1024 * 1024)

    def _get_cache_info_with_metadata(self, cache_dir: Path) -> Optional[dict]:
        """Get cache info with additional metadata like path and size."""
        info = self._load_cache_info(cache_dir)
        if info:
            info["cache_path"] = str(cache_dir)
            info["size_mb"] = self._calculate_cache_size_mb(cache_dir)
        return info

    def list_cached_frontends(self) -> list[dict]:
        """List all cached frontend PRs."""
        frontend_cache = self.cache_dir / "frontend"
        if not frontend_cache.exists():
            return []

        cached_prs = []
        for cache_dir in frontend_cache.iterdir():
            if not cache_dir.is_dir():
                continue
            info = self._get_cache_info_with_metadata(cache_dir)
            if info:
                cached_prs.append(info)

        return sorted(cached_prs, key=lambda x: x.get("cached_at", ""), reverse=True)

    def _is_cache_expired(self, cached_at: str) -> bool:
        """Check if a cache entry is expired based on its timestamp."""
        try:
            cache_time = datetime.fromisoformat(cached_at)
            return datetime.now() - cache_time > self.max_cache_age
        except (ValueError, TypeError):
            return True  # Consider invalid timestamps as expired

    def _get_expired_items(self, cached_items: list[dict]) -> list[dict]:
        """Get list of expired cache items."""
        expired = []
        for item in cached_items:
            cached_at = item.get("cached_at")
            if cached_at and self._is_cache_expired(cached_at):
                expired.append(item)
        return expired

    def _get_excess_items(self, cached_items: list[dict], expired_items: list[dict]) -> list[dict]:
        """Get list of items that exceed the maximum cache limit."""
        remaining_items = [item for item in cached_items if item not in expired_items]
        if len(remaining_items) > self.max_cache_items:
            # Return oldest items that exceed the limit
            return remaining_items[self.max_cache_items :]
        return []

    def _remove_cache_item(self, item: dict) -> None:
        """Remove a single cache item."""
        cache_path = Path(item["cache_path"])
        if cache_path.exists():
            pr_info = f"PR #{item.get('pr_number', '?')} ({item.get('pr_title', 'Unknown')[:30]}...)"
            rprint(f"[yellow]Removing old cache: {pr_info}[/yellow]")
            shutil.rmtree(cache_path)

    def enforce_cache_limits(self) -> None:
        """Remove old and excess cache entries to maintain limits."""
        cached_items = self.list_cached_frontends()

        # Get items to remove
        expired_items = self._get_expired_items(cached_items)
        excess_items = self._get_excess_items(cached_items, expired_items)

        # Remove all identified items
        items_to_remove = expired_items + excess_items
        for item in items_to_remove:
            self._remove_cache_item(item)

    def get_cache_age(self, cached_at: str) -> str:
        """Get human-readable age of cache entry"""
        try:
            cache_time = datetime.fromisoformat(cached_at)
            age = datetime.now() - cache_time

            if age.days > 0:
                return f"{age.days} day{'s' if age.days != 1 else ''} ago"
            if age.seconds > 3600:
                hours = age.seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            if age.seconds > 60:
                minutes = age.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            return "just now"
        except (json.JSONDecodeError, OSError):
            return "unknown"
