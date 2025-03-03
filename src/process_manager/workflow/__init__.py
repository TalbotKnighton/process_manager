# src/process_manager/workflow/__init__.py
"""Workflow package initialization."""
from process_manager.workflow.workflow_types import (
    ProcessConfig,
    ProcessMetadata,
    ProcessResult,
    ProcessState,
    ProcessType,
    RetryStrategy,
    WorkflowNode,
)
from process_manager.workflow.process import BaseProcess
from process_manager.workflow.core import Workflow, create_workflow

__all__ = [
    'Workflow',
    'create_workflow',
    'WorkflowNode',
    'BaseProcess',
    'ProcessConfig',
    'ProcessMetadata',
    'ProcessResult',
    'ProcessState',
    'ProcessType',
    'RetryStrategy',
]

# from process_manager.workflow.core import Workflow, create_workflow, WorkflowNode
# from process_manager.workflow.process import BaseProcess, ProcessConfig
# from process_manager.workflow.workflow_types import ProcessType, ProcessState, RetryStrategy
# from process_manager.workflow.implementations import CommandLineProcess, DataTransformProcess, AsyncAPIProcess

# __all__ = [
#     'Workflow',
#     'create_workflow',
#     'WorkflowNode',
#     'BaseProcess',
#     'ProcessConfig',
#     'ProcessType',
#     'ProcessState',
#     'RetryStrategy',
#     'CommandLineProcess',
#     'DataTransformProcess',
#     'AsyncAPIProcess',
# ]