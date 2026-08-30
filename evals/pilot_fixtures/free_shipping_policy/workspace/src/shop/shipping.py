"""Shipping cost calculation."""

from shop.policy import FREE_SHIPPING_THRESHOLD


def shipping_cost(order_total: float) -> float:
    if order_total >= FREE_SHIPPING_THRESHOLD:
        return 0.0
    return 5.99
