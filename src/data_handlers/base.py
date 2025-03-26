"""
Base module for named objects and their collections.
Provides common functionality for serialization and management of named objects.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Type, TypeVar, Dict
try:
    from typing import Self
except:
    from typing_extensions import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    SerializeAsAny,
    computed_field,
    model_validator,
)

class ObjectRegistry:
    """
    Generic registry for named object types.
    
    Stores object types for deserialization from JSON files.
    Types are stored at the class level for consistent access across modules.

    Types are stored at the class level for consistent access across modules.
    
    Methods:
        register: Register an object type
        get: Get an object type by name
        get_all: Get all registered object types
    """
    _registries: Dict[str, Dict[str, Type[NamedObject]]] = {}

    @classmethod
    def get_registry(cls, category: str) -> Dict[str, Type[NamedObject]]:
        """Get or create registry for a category."""
        # Get the actual string value if it's a PrivateAttr
        if hasattr(category, 'default'):
            category = category.default
        if category not in cls._registries:
            cls._registries[category] = {}
        return cls._registries[category]

    @classmethod
    def register(cls, category: str, obj_type: Type[NamedObject]) -> None:
        """Register an object type in its category."""
        # Get the actual string value if it's a PrivateAttr
        if hasattr(category, 'default'):
            category = category.default
        registry = cls.get_registry(category)
        registry[obj_type.__name__] = obj_type
    @classmethod
    def get(cls, category: str, name: str) -> Type[NamedObject]:
        """Get an object type by category and name."""
        registry = cls.get_registry(category)
        if name not in registry:
            raise ValueError(f"{name} not found in {category} registry")
        return registry[name]
    
    @classmethod
    def get_all(cls, category: str) -> list[Type[NamedObject]]:
        """Get all registered types for a category."""
        return list(cls.get_registry(category).values())

class NamedObject(BaseModel):
    """
    Base class for named objects with serialization support.
    
    Attributes:
        type (str): Name of object class type (computed field)
        name (str): Name of the object
        
    Configuration:
        model_config (ConfigDict): Pydantic model configuration
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    name: str
    _registry_category: str = "base"
    
    @model_validator(mode='before')
    @classmethod
    def _remove_computed_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Removes computed fields before validation."""
        for f in cls.model_computed_fields:
            data.pop(f, None)
        return data
    
    @computed_field
    @property
    def type(self) -> str:
        """Returns the name of the object type."""
        return type(self).__name__
    
    def __init_subclass__(cls, *, registry_category: str = None, **kwargs):
        """Register subclasses in appropriate registry category."""
        super().__init_subclass__(**kwargs)
        if registry_category:
            cls._registry_category = registry_category
        ObjectRegistry.register(cls._registry_category, cls)

NamedObjectType = TypeVar('NamedObjectType', bound=NamedObject)

class NamedObjectList(BaseModel):
    """
    List of named objects with type checking.
    
    Attributes:
        objects (list): List of named objects
        
    Example:
        ```python
        obj_list = NamedObjectList()
        obj_list.append(named_object)
        obj_list.extend([obj1, obj2, obj3])
        ```
    """
    model_config = ConfigDict(extra='forbid')
    
    objects: List[SerializeAsAny[InstanceOf[NamedObject]]] = Field(default_factory=list)
    _registry_category: str = "base"
    
    @classmethod
    def from_iterable(cls, iterable: Iterable[NamedObject]) -> Self:
        """
        Create instance from an iterable of named objects.
        
        Args:
            iterable (Iterable[NamedObject]): Objects to add to list
            
        Returns:
            Self: New instance containing the objects
        """
        return cls(objects=list(iterable))
    
    def append(self, obj: NamedObject) -> Self:
        """
        Append a single object to the list.
        
        Args:
            obj (NamedObject): Object to append
            
        Returns:
            Self: Returns self for chaining
        """
        self.objects.append(obj)
        return self
    
    def extend(self, objects: Iterable[NamedObject]) -> Self:
        """
        Extend list with multiple objects.
        
        Args:
            objects (Iterable[NamedObject]): Objects to add
            
        Returns:
            Self: Returns self for chaining
        """
        self.objects.extend(objects)
        return self
    
    def __len__(self) -> int:
        """Return number of objects in list."""
        return len(self.objects)
    
    def __iter__(self) -> Iterable[NamedObject]:
        """Iterate over objects in list."""
        return iter(self.objects)
    
    def __getitem__(self, idx: int) -> NamedObject:
        """Get object by index."""
        return self.objects[idx]

    def register_object(self, obj: NamedObject) -> Self:
        """
        Register a named object to the list with duplicate name checking.
        
        Args:
            obj (NamedObject): Object to register
            
        Returns:
            Self: Returns self for method chaining
            
        Raises:
            ValueError: If an object with the same name already exists
            
        Example:
            ```python
            obj_list = NamedObjectList()
            obj_list.register_object(NamedObject("x"))
            ```
        """
        # Check for duplicates
        for existing_obj in self.objects:
            if existing_obj.name == obj.name:
                raise ValueError(
                    f"Naming conflict: An object named '{obj.name}' already exists.\n"
                    f"\tExisting: \n{existing_obj.model_dump_json(indent=4)}\n"
                    f"\tNew: \n{obj.model_dump_json(indent=4)}"
                )
        self.objects.append(obj)
        return self

    def get_object(self, name: str) -> NamedObject:
        """
        Get a registered object by name.
        
        Args:
            name (str): Name of the object to retrieve
            
        Returns:
            NamedObject: The requested object
            
        Raises:
            KeyError: If no object exists with the given name
            
        Example:
            ```python
            obj_list = NamedObjectList()
            obj_list.append(NamedObject("x"))
            x = obj_list.get_object("x")
            ```
        """
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"No object found with name '{name}'")

    def get_objects(self) -> Iterable[NamedObject]:
        """
        Get all registered objects.
        
        Returns:
            Iterable[NamedObject]: Iterator over all stored objects
            
        Example:
            ```python
            obj_list = NamedObjectList()
            obj_list.extend([NamedObject("x"), NamedObject("y")])
            for obj in obj_list.get_objects():
                print(obj.name)
            ```
        """
        return iter(self.objects)

class NamedObjectHash(BaseModel):
    """
    Dictionary of named objects with type checking and conflict prevention.
    
    Attributes:
        objects (dict): Dictionary of named objects
    """
    objects: dict[str, SerializeAsAny[InstanceOf[NamedObject]]] = Field(default_factory=dict)
    _registry_category: str = "base"

    @model_validator(mode='before')
    @classmethod
    def deserialize_objects(cls, data: Any) -> Any:
        """Deserialize objects during validation."""
        if not isinstance(data, dict):
            return data
            
        objects = data.get('objects', {})
        if isinstance(objects, dict):
            for name, obj_data in objects.items():
                if isinstance(obj_data, dict):
                    type_name = obj_data.get('type')
                    if type_name:
                        # Remove type as it's not part of the constructor
                        obj_data = obj_data.copy()
                        obj_data.pop('type')
                        obj_type = ObjectRegistry.get(cls._registry_category, type_name)
                        data['objects'][name] = obj_type(**obj_data)
        return data

    def register_object(self, obj: NamedObject) -> Self:
        """
        Register a named object. Checks for naming conflicts.
        
        Args:
            obj (NamedObject): Object to register
            
        Raises:
            ValueError: If object with same name exists
        """
        if obj.name in self.objects:
            raise ValueError(
                f"Naming conflict: An object named '{obj.name}' already exists."
                f"\n\tExisting: \n{self.get_object(obj.name).model_dump_json(indent=4)}"
                f"\n\tNew: \n{obj.model_dump_json(indent=4)}"
            )
        self.objects[obj.name] = obj
        return self

    def get_object(self, name: str) -> NamedObject:
        """Get object by name."""
        return self.objects[name]
    
    def get_objects(self) -> Iterable[NamedObject]:
        """Get all objects."""
        return self.objects.values()
    
    def get_object_names(self) -> Iterable[str]:
        """Get names of all objects."""
        return self.objects.keys()