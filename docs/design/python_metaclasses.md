# Python Metaclasses Guide

## Core Concepts
1. A metaclass is a class for a class - it allows you to customize class creation
2. Metaclasses are called when a class is defined, not when it's instantiated
3. The metaclass can modify the class definition before it's created

## Key Rules

### 1. Inheritance Rules
- Metaclasses are inherited by subclasses
- If a class has a metaclass, all its subclasses must be compatible with that metaclass
- When there are multiple metaclasses in the inheritance hierarchy, they must be compatible

### 2. Creation Order
```python
class MyMetaclass(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # 1. __new__ is called first
        return super().__new__(mcs, name, bases, namespace)
    
    def __init__(cls, name, bases, namespace, **kwargs):
        # 2. __init__ is called after __new__
        super().__init__(name, bases, namespace)
    
    def __call__(cls, *args, **kwargs):
        # 3. __call__ is called when creating instances
        return super().__call__(*args, **kwargs)
```

### 3. Declaration Methods
```python
# Method 1: metaclass keyword
class MyClass(metaclass=MyMetaclass):
    pass

# Method 2: inheritance from a class with metaclass
class MyMetaclassBase(metaclass=MyMetaclass):
    pass
class MyClass(MyMetaclassBase):
    pass
```

### 4. Common Use Cases
```python
class RegisterMeta(type):
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        # Register all classes using this metaclass
        cls = super().__new__(mcs, name, bases, namespace)
        mcs.registry[name] = cls
        return cls

class ValidatorMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Add validation to all methods
        for key, value in namespace.items():
            if callable(value):
                namespace[key] = validate(value)
        return super().__new__(mcs, name, bases, namespace)
```

### 5. Working with Other Metaclasses
```python
# Combining metaclasses
class CombinedMeta(MetaclassA, MetaclassB):
    def __new__(mcs, name, bases, namespace):
        # Call both metaclass's __new__
        namespace = MetaclassA.__new__(mcs, name, bases, namespace)
        return MetaclassB.__new__(mcs, name, bases, namespace)
```

## Recommended References

### 1. Official Python Documentation
- [Python Data Model](https://docs.python.org/3/reference/datamodel.html#metaclasses)
- [Custom Metaclasses](https://docs.python.org/3/reference/datamodel.html#customizing-class-creation)

### 2. Books
- "Python in a Nutshell" by Alex Martelli (O'Reilly)
- "Fluent Python" by Luciano Ramalho (O'Reilly)

### 3. Real-world Example
```python
class RandomVariableMeta(type(BaseModel)):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Automatically apply squeezable decorator
        for method in ['sample', 'pdf', 'cdf']:
            if method in namespace and callable(namespace[method]):
                namespace[method] = squeezable(namespace[method])
        return super().__new__(mcs, name, bases, namespace, **kwargs)
```

## Common Gotchas
1. Multiple Inheritance: Be careful when combining classes with different metaclasses
2. Order of Operations: Remember metaclass code runs during class definition
3. Performance: Metaclasses can impact class creation performance
4. Complexity: They can make code harder to understand if overused

## Best Practices
1. Use metaclasses sparingly - only when class decorators or inheritance won't suffice
2. Document metaclass behavior clearly
3. Keep metaclass logic simple and focused
4. Consider alternatives like class decorators or descriptors first