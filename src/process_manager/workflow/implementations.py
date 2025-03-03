from .process import BaseProcess, ProcessConfig
from .workflow_types import ProcessType
import subprocess
import asyncio
from typing import Callable, Any
from .process import BaseProcess, ProcessConfig
from .workflow_types import ProcessType
import subprocess
import shlex
import json
from typing import Union, Dict, List
import tempfile
import os

class CommandLineProcess(BaseProcess):
    """
    Process implementation for running command-line programs.
    
    Supports several ways to handle input data:
    1. Template strings in command: "{var_name}"
    2. Environment variables
    3. Input files
    4. Command line arguments
    """
    
    def __init__(self, 
                 config: ProcessConfig, 
                 command: str,
                 input_mode: str = "template",
                 input_format: str = "json"):
        """
        Initialize CommandLineProcess.
        
        Args:
            config: Process configuration
            command: Command string to execute
            input_mode: How to pass input data to command
                       ("template", "env", "file", "args")
            input_format: Format for input data ("json", "text", "raw")
        """
        config.process_type = ProcessType.PROCESS
        super().__init__(config)
        self.command = command
        self.input_mode = input_mode
        self.input_format = input_format

    def _sync_execute(self, input_data: Any) -> Any:
        """
        Execute command with input data.
        
        Args:
            input_data: Data to be passed to the command
        
        Returns:
            Command output
        """
        try:
            formatted_command = self._prepare_command(input_data)
            env = self._prepare_environment(input_data)
            
            process = subprocess.run(
                formatted_command,
                shell=True,
                capture_output=True,
                text=True,
                env=env
            )
            
            if process.returncode != 0:
                raise Exception(f"Command failed: {process.stderr}")
            
            return process.stdout
            
        finally:
            # Cleanup any temporary files
            self._cleanup()

    def _prepare_command(self, input_data: Any) -> str:
        """Prepare command string based on input mode."""
        if self.input_mode == "template":
            # Replace template variables in command string
            if isinstance(input_data, dict):
                return self.command.format(**input_data)
            return self.command.format(input_data)
            
        elif self.input_mode == "args":
            # Add input data as command line arguments
            args = self._format_args(input_data)
            return f"{self.command} {args}"
            
        elif self.input_mode == "file":
            # Create temporary file with input data
            self.temp_file = self._create_temp_file(input_data)
            return f"{self.command} {self.temp_file.name}"
            
        else:  # env mode uses environment variables
            return self.command

    def _prepare_environment(self, input_data: Any) -> Dict[str, str]:
        """Prepare environment variables."""
        env = os.environ.copy()
        
        if self.input_mode == "env":
            # Add input data as environment variables
            if isinstance(input_data, dict):
                for key, value in input_data.items():
                    env[str(key)] = str(value)
            else:
                env["INPUT_DATA"] = str(input_data)
                
        return env

    def _format_args(self, input_data: Any) -> str:
        """Format input data as command line arguments."""
        if isinstance(input_data, dict):
            args = []
            for key, value in input_data.items():
                args.extend([f"--{key}", str(value)])
            return " ".join(shlex.quote(arg) for arg in args)
        return shlex.quote(str(input_data))

    def _create_temp_file(self, input_data: Any) -> tempfile.NamedTemporaryFile:
        """Create temporary file with input data."""
        temp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        
        if self.input_format == "json":
            json.dump(input_data, temp)
        else:
            temp.write(str(input_data))
            
        temp.close()
        return temp

    def _cleanup(self):
        """Clean up temporary resources."""
        if hasattr(self, 'temp_file'):
            os.unlink(self.temp_file.name)

class DataTransformProcess(BaseProcess):
    """Process implementation for data transformation."""
    
    def __init__(self, config: ProcessConfig, transform_func: Callable):
        config.process_type = ProcessType.PROCESS
        super().__init__(config)
        self.transform_func = transform_func

    def _sync_execute(self, input_data: Any) -> Any:
        return self.transform_func(input_data)

class AsyncAPIProcess(BaseProcess):
    """Process implementation for async API operations."""
    
    def __init__(self, config: ProcessConfig, url: str, method: str = "GET"):
        config.process_type = ProcessType.ASYNC
        super().__init__(config)
        self.url = url
        self.method = method

    async def execute(self, input_data: Any) -> Any:
        # Simulate API call
        await asyncio.sleep(1)
        return {"api_response": "data"}

class IOBoundProcess(BaseProcess):
    """Process for I/O-bound operations."""
    
    def __init__(self, config: ProcessConfig):
        config.process_type = ProcessType.THREAD  # Force thread pool
        super().__init__(config)

    def _sync_execute(self, input_data: Any) -> Any:
        # I/O bound work here
        result = None
        return result
