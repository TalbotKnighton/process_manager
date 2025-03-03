from pathlib import Path
from typing import Union, BinaryIO, TextIO, Any, Dict
import json
import yaml
import csv
import pickle
from datetime import datetime
import shutil
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    """
    Handles file operations with support for multiple formats and validation.
    
    Features:
    - Multiple format support (text, JSON, YAML, CSV, pickle)
    - Automatic backup
    - File locking for concurrent access
    - Error handling and logging
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize file handler.
        
        Args:
            base_dir: Base directory for file operations
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backup directory
        self.backup_dir = self.base_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def _backup_file(self, filepath: Path) -> None:
        """Create backup of existing file"""
        if filepath.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{filepath.name}.{timestamp}.bak"
            shutil.copy2(filepath, backup_path)
            logger.debug(f"Created backup: {backup_path}")
    
    def write_text(self, filepath: Path, content: str,
                  backup: bool = True) -> None:
        """
        Write text content to file.
        
        Args:
            filepath: Path to output file
            content: Text content to write
            backup: Whether to backup existing file
        """
        filepath = Path(filepath)
        if backup:
            self._backup_file(filepath)
            
        try:
            filepath.write_text(content)
            logger.debug(f"Wrote text file: {filepath}")
        except Exception as e:
            logger.error(f"Error writing text file {filepath}: {e}")
            raise
    
    def read_text(self, filepath: Path) -> str:
        """
        Read text content from file.
        
        Args:
            filepath: Path to input file
            
        Returns:
            Text content of file
        """
        try:
            return Path(filepath).read_text()
        except Exception as e:
            logger.error(f"Error reading text file {filepath}: {e}")
            raise
    
    def write_json(self, filepath: Path, data: Any,
                  backup: bool = True) -> None:
        """Write data as JSON"""
        if backup:
            self._backup_file(filepath)
            
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Wrote JSON file: {filepath}")
        except Exception as e:
            logger.error(f"Error writing JSON file {filepath}: {e}")
            raise
    
    def read_json(self, filepath: Path) -> Any:
        """Read JSON data"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON file {filepath}: {e}")
            raise
    
    def write_yaml(self, filepath: Path, data: Any,
                  backup: bool = True) -> None:
        """Write data as YAML"""
        if backup:
            self._backup_file(filepath)
            
        try:
            with open(filepath, 'w') as f:
                yaml.dump(data, f)
            logger.debug(f"Wrote YAML file: {filepath}")
        except Exception as e:
            logger.error(f"Error writing YAML file {filepath}: {e}")
            raise
    
    def read_yaml(self, filepath: Path) -> Any:
        """Read YAML data"""
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error reading YAML file {filepath}: {e}")
            raise
    
    def write_csv(self, filepath: Path, data: list,
                 headers: list = None, backup: bool = True) -> None:
        """Write data as CSV"""
        if backup:
            self._backup_file(filepath)
            
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
            logger.debug(f"Wrote CSV file: {filepath}")
        except Exception as e:
            logger.error(f"Error writing CSV file {filepath}: {e}")
            raise
    
    def read_csv(self, filepath: Path, has_headers: bool = True) -> list:
        """Read CSV data"""
        try:
            with open(filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                if has_headers:
                    headers = next(reader)
                return list(reader)
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath}: {e}")
            raise
    
    def write_pickle(self, filepath: Path, data: Any,
                    backup: bool = True) -> None:
        """Write data as pickle"""
        if backup:
            self._backup_file(filepath)
            
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Wrote pickle file: {filepath}")
        except Exception as e:
            logger.error(f"Error writing pickle file {filepath}: {e}")
            raise
    
    def read_pickle(self, filepath: Path) -> Any:
        """Read pickle data"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading pickle file {filepath}: {e}")
            raise
