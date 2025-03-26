from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from results_manager import (
    register_model, get_model_class, DEFAULT_NAMESPACE,
    get_namespaces, get_models_in_namespace,
    ResultsManager
)

# Example 1: Basic registration in default namespace
@register_model
class User(BaseModel):
    id: str
    name: str
    email: Optional[str] = None

# Example 2: Registration in a custom namespace
@register_model(namespace="analytics")
class Event(BaseModel):
    event_id: str
    timestamp: float
    user_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)

# Example 3: Alternative syntax for namespaces
@register_model("billing")
class Invoice(BaseModel):
    invoice_id: str
    amount: float
    paid: bool = False

def main():
    # Example 1: Basic retrieval from default namespace
    user_class = get_model_class("User")
    if user_class:
        user = user_class(id="123", name="John Doe", email="john@example.com")
        print(f"Created user: {user}")
    
    # Example 2: Retrieval from custom namespace
    event_class = get_model_class("Event", namespace="analytics")
    if event_class:
        event = event_class(
            event_id="evt_123",
            timestamp=1625097600,
            user_id="123",
            properties={"page": "home", "source": "direct"}
        )
        print(f"Created event: {event}")
    
    # Example 3: Working with multiple namespaces
    print("\nList of registered namespaces:")
    for ns in get_namespaces():
        print(f"Namespace: {ns}")
        models = get_models_in_namespace(ns)
        print(f"  Models: {', '.join(models)}")
    
    # Example 4: Using with ResultsManager
    results = ResultsManager("./example_data")
    
    # Store models from different namespaces
    invoice = Invoice(invoice_id="inv_001", amount=99.99)
    results.set("billing/invoices/inv_001", invoice)
    
    # For retrieval, you'd typically need to specify the model class
    # for models not in the default namespace
    retrieved_invoice = results.get("billing/invoices/inv_001", Invoice)
    print(f"\nRetrieved invoice: {retrieved_invoice}")
    
    # You can also register the same model in multiple namespaces
    # if you need it to be findable in both
    register_model(Invoice, namespace=DEFAULT_NAMESPACE)
    
    # Now it can be found in both namespaces
    assert get_model_class("Invoice", namespace="billing") is Invoice
    assert get_model_class("Invoice", namespace=DEFAULT_NAMESPACE) is Invoice

if __name__ == "__main__":
    main()