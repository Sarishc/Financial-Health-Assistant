"""
Caching utilities for the Financial Health Assistant
"""
import os
import json
import pickle
import hashlib
import time
import logging
from typing import Any, Dict, List, Callable, Optional, Union
from functools import wraps
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    """
    Class to manage in-memory and disk caching
    """
    
    def __init__(self, cache_dir: str = 'cache', max_memory_items: int = 100, 
                 default_ttl: int = 3600):
        """
        Initialize the cache manager
        
        Args:
            cache_dir: Directory to store cached files
            max_memory_items: Maximum number of items to keep in memory
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = cache_dir
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # In-memory cache
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        # Structure: {key: {'data': data, 'expires': timestamp}}
    
    def _get_cache_key(self, prefix: str, args: tuple, kwargs: Dict[str, Any]) -> str:
        """
        Generate a unique cache key from function arguments
        
        Args:
            prefix: Prefix for the key (usually function name)
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Unique cache key
        """
        # Convert arguments to a string representation
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
        # Create a hash of the arguments
        hash_input = f"{prefix}:{args_str}:{kwargs_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_disk_cache_path(self, key: str) -> str:
        """
        Get the file path for a disk cache item
        
        Args:
            key: Cache key
            
        Returns:
            File path for the cache item
        """
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        """
        Get an item from the in-memory cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached item if found and not expired, None otherwise
        """
        if key in self.memory_cache:
            cache_item = self.memory_cache[key]
            
            # Check if expired
            if cache_item['expires'] > time.time():
                logger.debug(f"Memory cache hit for key: {key}")
                return cache_item['data']
            else:
                # Remove expired item
                del self.memory_cache[key]
                logger.debug(f"Removed expired memory cache item: {key}")
        
        return None
    
    def get_from_disk(self, key: str) -> Optional[Any]:
        """
        Get an item from the disk cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached item if found and not expired, None otherwise
        """
        file_path = self._get_disk_cache_path(key)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    cache_item = pickle.load(f)
                
                # Check if expired
                if cache_item['expires'] > time.time():
                    logger.debug(f"Disk cache hit for key: {key}")
                    return cache_item['data']
                else:
                    # Remove expired item
                    os.remove(file_path)
                    logger.debug(f"Removed expired disk cache item: {key}")
            except Exception as e:
                logger.warning(f"Error reading disk cache: {str(e)}")
                # Remove corrupted cache file
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        return None
    
    def set_in_memory(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Store an item in the in-memory cache
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (default: use default_ttl)
        """
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl
        
        # Calculate expiration time
        expires = time.time() + ttl
        
        # Add to cache
        self.memory_cache[key] = {
            'data': data,
            'expires': expires
        }
        
        # Check if we need to clear space
        if len(self.memory_cache) > self.max_memory_items:
            self._cleanup_memory_cache()
        
        logger.debug(f"Added item to memory cache: {key}")
    
    def set_on_disk(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Store an item in the disk cache
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (default: use default_ttl)
        """
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl
        
        # Calculate expiration time
        expires = time.time() + ttl
        
        # Create cache item
        cache_item = {
            'data': data,
            'expires': expires
        }
        
        # Save to disk
        file_path = self._get_disk_cache_path(key)
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(cache_item, f)
            logger.debug(f"Added item to disk cache: {key}")
        except Exception as e:
            logger.warning(f"Error writing to disk cache: {str(e)}")
    
    def _cleanup_memory_cache(self) -> None:
        """
        Remove expired items and reduce cache size if needed
        """
        current_time = time.time()
        
        # First, remove expired items
        expired_keys = [
            key for key, item in self.memory_cache.items()
            if item['expires'] <= current_time
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # If still too many items, remove the oldest ones
        if len(self.memory_cache) > self.max_memory_items:
            # Sort by expiration time (oldest first)
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1]['expires']
            )
            
            # Remove oldest items
            items_to_remove = len(self.memory_cache) - self.max_memory_items
            for key, _ in sorted_items[:items_to_remove]:
                del self.memory_cache[key]
        
        logger.debug(f"Cleaned up memory cache. Current size: {len(self.memory_cache)}")
    
    def clear_expired(self) -> None:
        """
        Clear all expired items from memory and disk cache
        """
        # Clear expired memory items
        current_time = time.time()
        expired_keys = [
            key for key, item in self.memory_cache.items()
            if item['expires'] <= current_time
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Clear expired disk items
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(file_path, 'rb') as f:
                        cache_item = pickle.load(f)
                    
                    if cache_item['expires'] <= current_time:
                        os.remove(file_path)
                        logger.debug(f"Removed expired disk cache item: {filename}")
                except Exception as e:
                    logger.warning(f"Error checking disk cache expiration: {str(e)}")
                    # Remove corrupted cache file
                    os.remove(file_path)
    
    def clear_all(self) -> None:
        """
        Clear all cache items (memory and disk)
        """
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, filename))
        
        logger.info("Cleared all caches")
    
    def memoize(self, ttl: Optional[int] = None, use_disk: bool = False):
        """
        Decorator to cache function results
        
        Args:
            ttl: Time-to-live in seconds (default: use default_ttl)
            use_disk: Whether to use disk caching
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                key = self._get_cache_key(func.__name__, args, kwargs)
                
                # Try to get from memory cache first
                cached_result = self.get_from_memory(key)
                if cached_result is not None:
                    return cached_result
                
                # If using disk cache, try to get from disk
                if use_disk:
                    cached_result = self.get_from_disk(key)
                    if cached_result is not None:
                        # Also store in memory for faster access next time
                        self.set_in_memory(key, cached_result, ttl)
                        return cached_result
                
                # If not found in cache, call the function
                result = func(*args, **kwargs)
                
                # Store in memory cache
                self.set_in_memory(key, result, ttl)
                
                # Also store in disk cache if requested
                if use_disk:
                    self.set_on_disk(key, result, ttl)
                
                return result
            
            return wrapper
        
        return decorator

# Create a global cache manager instance
cache_manager = CacheManager()

# Convenience decorators
def cache_in_memory(ttl: Optional[int] = None):
    """
    Decorator to cache function results in memory
    
    Args:
        ttl: Time-to-live in seconds
        
    Returns:
        Decorated function
    """
    return cache_manager.memoize(ttl=ttl, use_disk=False)

def cache_on_disk(ttl: Optional[int] = None):
    """
    Decorator to cache function results on disk
    
    Args:
        ttl: Time-to-live in seconds
        
    Returns:
        Decorated function
    """
    return cache_manager.memoize(ttl=ttl, use_disk=True)