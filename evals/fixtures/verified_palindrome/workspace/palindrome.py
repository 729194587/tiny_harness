"""Palindrome predicate."""


def is_palindrome(value: str) -> bool:
    return value == value[::-1]
