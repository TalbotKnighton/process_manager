from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Union, Optional, TypeVar, Generic
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from results_manager.model_registry import DEFAULT_NAMESPACE
from results_manager.backends.base import SetBehavior

T = TypeVar('T', bound=BaseModel)

class AsyncResultsBackend(Generic[T], ABC):
    """
    Abstract base class for async results storage backends.
    
    Implementations should provide asynchronous storage and retrieval of Pydantic models.
    """
    
    @abstractmethod
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
            
        Raises:
            FileExistsError: If data already exists (for RAISE_IF_EXISTS) or
                             if different data exists (for RAISE_IF_DIFFERENT)
        """
        pass
    
    @abstractmethod
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
            
        Raises:
            FileNotFoundError: If the result doesn't exist
            ValueError: If the model type is not registered
        """
        pass
    
    @abstractmethod
    async def exists(self, result_id: Union[str, List[str]]) -> bool:
        """
        Asynchronously check if a result exists for the given ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if the result exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_ids(self, prefix: Union[str, List[str]] = None) -> List[str]:
        """
        Asynchronously list all result IDs, optionally filtered by a prefix.
        
        Args:
            prefix: Optional prefix path to filter results
            
        Returns:
            List of result IDs
        """
        pass
    
    @abstractmethod
    async def delete(self, result_id: Union[str, List[str]]) -> bool:
        """
        Asynchronously delete a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """
        Asynchronously clear all stored results.
        """
        pass