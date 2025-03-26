from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Union, Optional, TypeVar, Generic
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class SetBehavior(Enum):
    """
    Defines behavior when setting data for an ID that already exists.
    """
    RAISE_IF_EXISTS = "raise_if_exists"  # Raise error if ID already exists
    RAISE_IF_DIFFERENT = "raise_if_different"  # Raise error if data exists AND is different
    OVERWRITE = "overwrite"  # Always overwrite existing data
    SKIP_IF_EXISTS = "skip_if_exists"  # Do nothing if data already exists

class ResultsBackend(Generic[T], ABC):
    """
    Abstract base class for results storage backends.
    
    Implementations should provide storage and retrieval of Pydantic models
    based on unique IDs.
    """
    
    @abstractmethod
    def set(self, 
            result_id: Union[str, List[str]], 
            data: BaseModel, 
            behavior: SetBehavior = SetBehavior.RAISE_IF_EXISTS,
            namespace: Optional[str] = None,
            strict_namespace: bool = False) -> bool:
        """
        Store a result with the given ID.
        
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
    def get(self, 
            result_id: Union[str, List[str]], 
            model_class: Optional[Type[T]] = None,
            namespace: Optional[str] = None) -> T:
        """
        Retrieve a result by ID.
        
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
    def exists(self, result_id: Union[str, List[str]]) -> bool:
        """
        Check if a result exists for the given ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if the result exists, False otherwise
        """
        pass
    
    @abstractmethod
    def list_ids(self, prefix: Union[str, List[str]] = None) -> List[str]:
        """
        List all result IDs, optionally filtered by a prefix.
        
        Args:
            prefix: Optional prefix path to filter results
            
        Returns:
            List of result IDs
        """
        pass
    
    @abstractmethod
    def delete(self, result_id: Union[str, List[str]]) -> bool:
        """
        Delete a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all stored results.
        """
        pass