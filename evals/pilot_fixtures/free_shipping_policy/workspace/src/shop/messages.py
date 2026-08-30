"""Customer-facing policy messages."""

from shop.policy import FREE_SHIPPING_THRESHOLD


def free_shipping_message() -> str:
    return (
        f"Free shipping on orders of ${FREE_SHIPPING_THRESHOLD:.2f} or more."
    )
