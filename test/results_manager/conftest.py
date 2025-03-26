import pytest
import tempfile
import shutil
from pathlib import Path

from pydantic import BaseModel
from typing import Optional, List

from pydantic_workflow.results_manager import ResultsManager, register_model, clear_registry

# Clear the registry at the start to ensure a clean state
clear_registry()

# Test models
@register_model
class TestPerson(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


@register_model
class TestTask(BaseModel):
    task_id: str
    status: str
    value: float
    metadata: Optional[dict] = None


@register_model
class TestNestedModel(BaseModel):
    id: str
    items: List[TestPerson]


@pytest.fixture(scope="session", autouse=True)
def register_test_models():
    """Ensure test models are registered."""
    # The models are already registered via decorators,
    # but this fixture ensures they are imported and
    # available for all tests
    return


@pytest.fixture
def temp_dir():
    """Provides a temporary directory for test data."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def results_manager(temp_dir):
    """Provides a ResultsManager instance with a temporary directory."""
    return ResultsManager(temp_dir)


@pytest.fixture
def sample_person():
    """Returns a sample TestPerson model."""
    return TestPerson(name="John Doe", age=30, email="john@example.com")


@pytest.fixture
def different_person():
    """Returns a different TestPerson model."""
    return TestPerson(name="Jane Smith", age=25, email="jane@example.com")


@pytest.fixture
def same_data_different_values():
    """Returns a TestPerson with same structure but different values."""
    return TestPerson(name="John Doe", age=31, email="john.doe@example.com")


@pytest.fixture
def sample_task():
    """Returns a sample TestTask model."""
    return TestTask(
        task_id="task123",
        status="completed",
        value=0.95,
        metadata={"source": "unit-test"}
    )


@pytest.fixture
def nested_model(sample_person, different_person):
    """Returns a nested model containing other models."""
    return TestNestedModel(
        id="nested1",
        items=[sample_person, different_person]
    )