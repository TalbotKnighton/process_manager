import json
from pathlib import Path
from typing import Dict, Any, List, Type, Union, Optional, TypeVar, Generic
import os
import aiosqlite

from pydantic import BaseModel

from results_manager.async_backends.base import AsyncResultsBackend
from results_manager.backends.base import SetBehavior
from results_manager.model_registry import (
    get_model_class, DEFAULT_NAMESPACE, find_model_namespace,
    find_model_in_all_namespaces
)

T = TypeVar('T', bound=BaseModel)

class AsyncSqliteBackend(AsyncResultsBackend[T]):
    """
    Async SQLite-based implementation of AsyncResultsBackend.
    
    Uses aiosqlite for asynchronous database operations.
    """
    
    def __init__(self, db_path: Union[str, Path], create_if_missing: bool = True):
        """
        Initialize the AsyncSqliteBackend.
        
        Args:
            db_path: Path to the SQLite database file
            create_if_missing: Whether to create the database if it doesn't exist
        """
        self.db_path = Path(db_path)
        
        # Check if the directory exists
        if not self.db_path.parent.exists():
            if create_if_missing:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directory for database {self.db_path.parent} does not exist")
    
    async def _init_db(self):
        """Initialize the database and create tables if they don't exist."""
        async with aiosqlite.connect(str(self.db_path)) as conn:
            # Create results table
            await conn.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                namespace TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create index on model_type and namespace
            await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_results_model_type_namespace 
            ON results (model_type, namespace)
            ''')
            
            # Create index for prefix queries
            await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_results_id 
            ON results (id)
            ''')
            
            await conn.commit()
    
    def _normalize_id(self, result_id: Union[str, List[str]]) -> str:
        """
        Convert a result ID or hierarchical tags into a normalized string.
        
        Args:
            result_id: Single string ID or list of hierarchical tags
            
        Returns:
            Normalized string ID
        """
        if isinstance(result_id, str):
            return result_id
        else:
            # Join list with slashes
            return '/'.join([part for part in result_id if part])
    
    def _are_models_equal(self, model1: BaseModel, model2: BaseModel) -> bool:
        """
        Compare two models for equality based on their dictionary representation.
        
        Args:
            model1: First model to compare
            model2: Second model to compare
            
        Returns:
            True if models have the same data content, False otherwise
        """
        return model1.model_dump() == model2.model_dump()
    
    async def set(self, 
                 result_id: Union[str, List[str]], 
                 data: BaseModel, 
                 behavior: SetBehavior = SetBehavior.RAISE_IF_EXISTS,
                 namespace: Optional[str] = None,
                 strict_namespace: bool = False) -> bool:
        """
        Asynchronously store a result with the given ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            data: Pydantic model instance to store
            behavior: How to handle existing data with the same ID
            namespace: Optional namespace to store the model in
            strict_namespace: If True, raises an error if model is in multiple namespaces
            
        Returns:
            True if data was written, False if skipped
        """
        # Ensure database is initialized
        await self._init_db()
        
        normalized_id = self._normalize_id(result_id)
        
        # Determine the namespace to use
        if namespace is None:
            # Try to find the namespace from the model class
            try:
                model_namespace = find_model_namespace(data.__class__, strict=strict_namespace)
                if model_namespace is not None:
                    namespace = model_namespace
                else:
                    namespace = DEFAULT_NAMESPACE
            except ValueError as e:
                # Re-raise the error about multiple namespaces
                raise ValueError(
                    f"Cannot automatically determine namespace for {data.__class__.__name__} "
                    f"when saving to '{normalized_id}': {str(e)}"
                ) from e
        
        async with aiosqlite.connect(str(self.db_path)) as conn:
            # Check if entry already exists
            async with conn.execute(
                "SELECT model_type, namespace, data FROM results WHERE id = ?", 
                (normalized_id,)
            ) as cursor:
                existing = await cursor.fetchone()
            
            if existing:
                if behavior == SetBehavior.RAISE_IF_EXISTS:
                    raise FileExistsError(f"Data already exists for ID: {normalized_id}")
                
                elif behavior == SetBehavior.RAISE_IF_DIFFERENT or behavior == SetBehavior.SKIP_IF_EXISTS:
                    # Compare data directly
                    stored_model_type, stored_namespace, stored_data_json = existing
                    stored_data = json.loads(stored_data_json)
                    
                    if stored_model_type == data.__class__.__name__:
                        # Direct comparison if same model type
                        if behavior == SetBehavior.SKIP_IF_EXISTS:
                            # If the data is the same, skip
                            if stored_data == data.model_dump():
                                return False
                        elif behavior == SetBehavior.RAISE_IF_DIFFERENT:
                            # If the data is different, raise
                            if stored_data != data.model_dump():
                                raise FileExistsError(f"Different data already exists for ID: {normalized_id}")
            
            # Prepare data for storage
            model_type = data.__class__.__name__
            serialized_data = json.dumps(data.model_dump())
            
            # Insert or update the record
            await conn.execute('''
            INSERT OR REPLACE INTO results (id, model_type, namespace, data, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (normalized_id, model_type, namespace, serialized_data))
            
            await conn.commit()
            return True
    
    async def get(self, 
                 result_id: Union[str, List[str]], 
                 model_class: Optional[Type[T]] = None,
                 namespace: Optional[str] = None) -> T:
        """
        Asynchronously retrieve a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            model_class: Optional model class to validate against
            namespace: Optional namespace to look in
            
        Returns:
            Pydantic model instance
        """
        # Ensure database is initialized
        await self._init_db()
        
        normalized_id = self._normalize_id(result_id)
        
        async with aiosqlite.connect(str(self.db_path)) as conn:
            # Query the database
            async with conn.execute(
                "SELECT model_type, namespace, data FROM results WHERE id = ?", 
                (normalized_id,)
            ) as cursor:
                result = await cursor.fetchone()
            
            if not result:
                raise FileNotFoundError(f"No result found for ID: {normalized_id}")
            
            model_type_name, stored_namespace, data_json = result
            
            # Check for missing model_type
            if not model_type_name:
                raise ValueError(f"Stored data missing model type information")
            
            # Parse the JSON data
            stored_data = json.loads(data_json)
            
            # If no model class is provided, try to find it from the registry
            if not model_class:
                # Use the stored namespace if none provided
                lookup_namespace = namespace if namespace is not None else stored_namespace
                
                model_class = get_model_class(model_type_name, namespace=lookup_namespace)
                
                # If not found in the specified namespace, try alternatives
                if not model_class:
                    # Try finding in all namespaces
                    model_matches = find_model_in_all_namespaces(model_type_name)
                    if model_matches:
                        # Use the first match
                        first_namespace, model_class = model_matches[0]
                    else:
                        namespaces_tried = [lookup_namespace]
                        if lookup_namespace != DEFAULT_NAMESPACE:
                            namespaces_tried.append(DEFAULT_NAMESPACE)
                        
                        raise ValueError(
                            f"Model type '{model_type_name}' is not registered in "
                            f"namespace '{lookup_namespace}' or any other namespace. "
                            f"Tried namespaces: {', '.join(namespaces_tried)}"
                        )
            
            # Validate and return the model instance
            return model_class.model_validate(stored_data)
    
    async def exists(self, result_id: Union[str, List[str]]) -> bool:
        """
        Asynchronously check if a result exists for the given ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if the result exists, False otherwise
        """
        # Ensure database is initialized
        await self._init_db()
        
        normalized_id = self._normalize_id(result_id)
        
        async with aiosqlite.connect(str(self.db_path)) as conn:
            # Check if the ID exists
            async with conn.execute(
                "SELECT 1 FROM results WHERE id = ? LIMIT 1", 
                (normalized_id,)
            ) as cursor:
                result = await cursor.fetchone()
            
            return result is not None
    
    async def list_ids(self, prefix: Union[str, List[str]] = None) -> List[str]:
        """
        Asynchronously list all result IDs, optionally filtered by a prefix.
        
        Args:
            prefix: Optional prefix path to filter results
            
        Returns:
            List of result IDs
        """
        # Ensure database is initialized
        await self._init_db()
        
        async with aiosqlite.connect(str(self.db_path)) as conn:
            if prefix is None:
                # Get all IDs
                async with conn.execute("SELECT id FROM results ORDER BY id") as cursor:
                    rows = await cursor.fetchall()
            else:
                # Get IDs matching the prefix
                normalized_prefix = self._normalize_id(prefix)
                query_prefix = f"{normalized_prefix}%" if normalized_prefix else "%"
                async with conn.execute(
                    "SELECT id FROM results WHERE id LIKE ? ORDER BY id", 
                    (query_prefix,)
                ) as cursor:
                    rows = await cursor.fetchall()
            
            # Extract and return the IDs
            return [row[0] for row in rows]
    
    async def delete(self, result_id: Union[str, List[str]]) -> bool:
        """
        Asynchronously delete a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if deleted, False if not found
        """
        # Ensure database is initialized
        await self._init_db()
        
        normalized_id = self._normalize_id(result_id)
        
        # Check if it exists first
        if not await self.exists(normalized_id):
            return False
        
        async with aiosqlite.connect(str(self.db_path)) as conn:
            # Delete the record
            await conn.execute("DELETE FROM results WHERE id = ?", (normalized_id,))
            await conn.commit()
            
            return True
    
    async def clear(self) -> None:
        """
        Asynchronously clear all stored results.
        """
        # Make sure the database directory exists
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            return  # No database yet, so nothing to clear
        
        try:
            # Initialize the database if needed
            await self._init_db()
            
            async with aiosqlite.connect(str(self.db_path)) as conn:
                # Delete all records
                await conn.execute("DELETE FROM results")
                await conn.commit()
                
        except Exception as e:
            # Log the error or handle it as appropriate
            print(f"Error clearing SQLite database: {e}")
            raise