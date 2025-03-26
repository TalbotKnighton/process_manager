import json
from pathlib import Path
from typing import List, Type, Union, Optional, TypeVar
import shutil
import tempfile
import hashlib
from filelock import FileLock

from pydantic import BaseModel

from results_manager.backends.base import ResultsBackend, SetBehavior
from results_manager.model_registry import (
    get_model_class, DEFAULT_NAMESPACE, find_model_namespace,
    find_model_in_all_namespaces
)

T = TypeVar('T', bound=BaseModel)

class FileBackend(ResultsBackend[T]):
    """
    File-based implementation of ResultsBackend.
    
    Stores results as JSON files in a hierarchical directory structure.
    """
    
    def __init__(self, base_dir: Union[str, Path], create_if_missing: bool = True, locks_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the FileBackend.
        
        Args:
            base_dir: Base directory to store results
            create_if_missing: Whether to create the directory if it doesn't exist
            locks_dir: Directory to store lock files. If None, uses a system temp directory.
        """
        self.base_dir = Path(base_dir)
        
        if create_if_missing and not self.base_dir.exists():
            self.base_dir.mkdir(parents=True)
        elif not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory {self.base_dir} does not exist")
            
        # Set up locks directory
        if locks_dir is None:
            self.locks_dir = Path(tempfile.gettempdir()) / "results_manager_locks"
        else:
            self.locks_dir = Path(locks_dir)
            
        # Create locks directory if it doesn't exist
        if not self.locks_dir.exists():
            self.locks_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path_from_id(self, result_id: Union[str, List[str]]) -> Path:
        """
        Convert a result ID or hierarchical tags into a file path.
        
        Args:
            result_id: Single string ID or list of hierarchical tags
            
        Returns:
            Path to the JSON file for this result
        """
        if isinstance(result_id, str):
            path_parts = result_id.split('/')
        else:
            path_parts = result_id
            
        # Ensure we have valid path parts
        path_parts = [part for part in path_parts if part]
        
        if not path_parts:
            raise ValueError("Invalid result ID: empty ID")
            
        # The last part is the filename
        filename = f"{path_parts[-1]}.json"
        directory = self.base_dir.joinpath(*path_parts[:-1]) if len(path_parts) > 1 else self.base_dir
        
        return directory / filename
    
    def _get_lock_path(self, file_path: Path) -> Path:
        """
        Get the path to the lock file for a given result file.
        
        Args:
            file_path: Path to the result file
            
        Returns:
            Path to the lock file
        """
        # Create a safe filename for the lock file using a hash
        rel_path = str(file_path.relative_to(self.base_dir))
        hash_name = hashlib.md5(rel_path.encode()).hexdigest()
        return self.locks_dir / f"{hash_name}.lock"
    
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
        """
        file_path = self._get_path_from_id(result_id)
        lock_path = self._get_lock_path(file_path)
        
        # Use file lock with a timeout to avoid deadlocks
        with FileLock(lock_path, timeout=10):  # 10 second timeout
            # Handle existing data according to behavior
            if file_path.exists():
                if behavior == SetBehavior.RAISE_IF_EXISTS:
                    raise FileExistsError(f"Data already exists for ID: {result_id}")
                
                elif behavior == SetBehavior.SKIP_IF_EXISTS:
                    try:
                        # Simplified logic for SKIP_IF_EXISTS
                        with open(file_path, 'r') as f:
                            stored_data = json.load(f)
                        
                        # Compare model types
                        if stored_data.get("model_type") == data.__class__.__name__:
                            # Direct comparison of dumped data
                            if stored_data.get("data") == data.model_dump():
                                return False  # Skip if exactly the same
                    except (json.JSONDecodeError, KeyError, FileNotFoundError):
                        # If any error occurs during comparison, default to overwriting
                        pass
                
                elif behavior == SetBehavior.RAISE_IF_DIFFERENT:
                    try:
                        # Load existing data for comparison
                        with open(file_path, 'r') as f:
                            stored_data = json.load(f)
                        
                        # Compare model types
                        if stored_data.get("model_type") == data.__class__.__name__:
                            # Direct comparison of dumped data
                            if stored_data.get("data") != data.model_dump():
                                raise FileExistsError(f"Different data already exists for ID: {result_id}")
                    except (json.JSONDecodeError, KeyError, FileNotFoundError):
                        # If we can't load the file properly, treat as different
                        raise FileExistsError(f"Invalid data exists for ID: {result_id}")
            
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
                        f"when saving to '{result_id}': {str(e)}"
                    ) from e
            
            # Ensure the directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Store the model type and namespace along with the data
            serialized_data = {
                "model_type": data.__class__.__name__,
                "namespace": namespace,
                "data": data.model_dump()
            }
            
            # Use atomic write pattern for extra safety
            temp_file = file_path.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(serialized_data, f, indent=2)
                
            # Rename is atomic on most filesystems
            temp_file.replace(file_path)
                
            return True
        
    def get(self, 
            result_id: Union[str, List[str]], 
            model_class: Optional[Type[T]] = None,
            namespace: Optional[str] = None) -> T:
        """
        Retrieve a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            model_class: Optional model class to validate against. If not provided,
                         the stored model type will be used.
            namespace: Optional namespace override to look for the model in
                          
        Returns:
            Pydantic model instance
            
        Raises:
            FileNotFoundError: If the result doesn't exist
            ValueError: If the model type is not registered
            ValidationError: If the data doesn't match the model schema
        """
        file_path = self._get_path_from_id(result_id)
        lock_path = self._get_lock_path(file_path)
        
        # Use file lock to ensure thread/process safety
        with FileLock(lock_path):
            if not file_path.exists():
                raise FileNotFoundError(f"No result found for ID: {result_id}")
            
            with open(file_path, 'r') as f:
                stored_data = json.load(f)
            
            # Check for missing model_type even when model_class is provided
            model_type_name = stored_data.get("model_type")
            if not model_type_name:
                raise ValueError(f"Stored data missing model type information")
            
            # If no model class is provided, try to find it from the registry
            if not model_class:
                # Use the stored namespace if none provided
                stored_namespace = stored_data.get("namespace", DEFAULT_NAMESPACE)
                lookup_namespace = namespace if namespace is not None else stored_namespace
                
                model_class = get_model_class(model_type_name, namespace=lookup_namespace)
                
                # If not found in the specified namespace, try alternatives
                # Continue from where we left off:

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
            
            # Get the data to validate outside the lock
            data = stored_data["data"]
        
        # Validate outside the lock to minimize lock time
        return model_class.model_validate(data)
    
    def exists(self, result_id: Union[str, List[str]]) -> bool:
        """
        Check if a result exists for the given ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if the result exists, False otherwise
        """
        file_path = self._get_path_from_id(result_id)
        lock_path = self._get_lock_path(file_path)
        
        # Use file lock to ensure consistent state
        with FileLock(lock_path):
            return file_path.exists()
    
    def list_ids(self, prefix: Union[str, List[str]] = None) -> List[str]:
        """
        List all result IDs, optionally filtered by a prefix.
        
        Args:
            prefix: Optional prefix path to filter results
            
        Returns:
            List of result IDs
        """
        if prefix is None:
            base_path = self.base_dir
        else:
            if isinstance(prefix, str):
                prefix = prefix.split('/')
            base_path = self.base_dir.joinpath(*prefix)
            
        if not base_path.exists():
            return []
            
        result_ids = []
        # No need for locking as we're just reading directory structure
        for path in base_path.rglob("*.json"):
            # Convert path to relative path from base_dir
            rel_path = path.relative_to(self.base_dir)
            # Remove .json extension and convert to string
            result_id = str(rel_path.with_suffix(''))
            result_ids.append(result_id)
            
        return result_ids
    
    def delete(self, result_id: Union[str, List[str]]) -> bool:
        """
        Delete a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_path_from_id(result_id)
        lock_path = self._get_lock_path(file_path)
        
        # Use file lock to ensure thread/process safety
        with FileLock(lock_path):
            if not file_path.exists():
                return False
                
            file_path.unlink()
            
            # Try to clean up empty directories
            current_dir = file_path.parent
            while current_dir != self.base_dir:
                if not any(current_dir.iterdir()):
                    current_dir.rmdir()
                    current_dir = current_dir.parent
                else:
                    break
                    
            return True
    
    def clear(self) -> None:
        """
        Clear all stored results.
        """
        # For clear(), we'll use a more aggressive approach of deleting then recreating
        # the directory, which avoids having to lock individual files
        if self.base_dir.exists():
            # Create a temporary lock file for the entire directory
            lock_path = self.locks_dir / "clear_all.lock"
            with FileLock(lock_path):
                # Save the path
                path = self.base_dir
                # Delete everything
                shutil.rmtree(str(self.base_dir))
                # Recreate the directory
                self.base_dir.mkdir(parents=True)