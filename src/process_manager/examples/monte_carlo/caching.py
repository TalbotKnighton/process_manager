from pathlib import Path
import hashlib
import json
import shutil
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

class SimulationCache:
    """
    Manages caching of simulation results to avoid redundant calculations.
    
    This cache system provides:
    - Content-based caching using parameter hashing
    - Cache expiration
    - Cache statistics
    - Disk space management
    
    Attributes:
        cache_dir: Directory for cached results
        max_cache_size: Maximum cache size in bytes
        cache_ttl: Time-to-live for cached results
    """
    
    def __init__(self,
                 cache_dir: Path,
                 max_cache_size: int = 1024**3,  # 1GB default
                 cache_ttl: timedelta = timedelta(days=7)):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        
        # Initialize cache
        self._init_cache()
    
    def _init_cache(self):
        """Initialize cache and clean expired entries"""
        self._clean_expired()
        self._ensure_size_limit()
    
    def _compute_hash(self, params: Dict[str, Any]) -> str:
        """Compute stable hash for parameter set"""
        # Sort keys for stable hash
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()
    
    def _get_cache_path(self, param_hash: str) -> Path:
        """Get cache file path for parameter hash"""
        return self.cache_dir / f"{param_hash}.cache"
    
    def _get_metadata_path(self, param_hash: str) -> Path:
        """Get metadata file path for cache entry"""
        return self.cache_dir / f"{param_hash}.meta"
    
    def _save_metadata(self, param_hash: str, params: Dict[str, Any]):
        """Save metadata for cache entry"""
        metadata = {
            "created": datetime.now().isoformat(),
            "params": params,
            "hash": param_hash
        }
        with open(self._get_metadata_path(param_hash), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _is_expired(self, param_hash: str) -> bool:
        """Check if cache entry has expired"""
        meta_path = self._get_metadata_path(param_hash)
        if not meta_path.exists():
            return True
            
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        created = datetime.fromisoformat(metadata["created"])
        return datetime.now() - created > self.cache_ttl
    
    def _clean_expired(self):
        """Remove expired cache entries"""
        for meta_file in self.cache_dir.glob("*.meta"):
            param_hash = meta_file.stem
            if self._is_expired(param_hash):
                self._remove_cache_entry(param_hash)
    
    def _get_cache_size(self) -> int:
        """Get total size of cache in bytes"""
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.*"))
    
    def _ensure_size_limit(self):
        """Ensure cache size is within limits"""
        while self._get_cache_size() > self.max_cache_size:
            # Remove oldest entries first
            entries = []
            for meta_file in self.cache_dir.glob("*.meta"):
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                created = datetime.fromisoformat(metadata["created"])
                entries.append((created, meta_file.stem))
            
            if not entries:
                break
                
            # Remove oldest entry
            entries.sort()
            self._remove_cache_entry(entries[0][1])
    
    def _remove_cache_entry(self, param_hash: str):
        """Remove cache entry and its metadata"""
        cache_path = self._get_cache_path(param_hash)
        meta_path = self._get_metadata_path(param_hash)
        
        if cache_path.exists():
            cache_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
    
    def get(self, params: Dict[str, Any]) -> Optional[Path]:
        """
        Retrieve cached result for parameter set.
        
        Args:
            params: Simulation parameters
            
        Returns:
            Path to cached result file or None if not found
        """
        param_hash = self._compute_hash(params)
        cache_path = self._get_cache_path(param_hash)
        
        if cache_path.exists() and not self._is_expired(param_hash):
            self.hits += 1
            return cache_path
        
        self.misses += 1
        return None
    
    def put(self, params: Dict[str, Any], result_file: Path):
        """
        Cache result file for parameter set.
        
        Args:
            params: Simulation parameters
            result_file: Path to result file to cache
        """
        param_hash = self._compute_hash(params)
        cache_path = self._get_cache_path(param_hash)
        
        # Copy result to cache
        shutil.copy2(result_file, cache_path)
        self._save_metadata(param_hash, params)
        
        # Maintain cache size limit
        self._ensure_size_limit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "size_bytes": self._get_cache_size(),
            "size_limit_bytes": self.max_cache_size,
            "entry_count": len(list(self.cache_dir.glob("*.cache")))
        }
