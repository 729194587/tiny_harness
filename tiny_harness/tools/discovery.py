"""Convention-based discovery for TinyHarness Tool modules."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from tiny_harness.tools.definition import ToolDefinition
from tiny_harness.tools.registry import ToolRegistry


TOOL_FACTORY_NAME = "build_tools"


def discover_tools(
    context: Any,
    *,
    package_name: str = "tiny_harness.tools",
) -> ToolRegistry:
    """Discover and construct definitions exported by modules in a package."""

    package = importlib.import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        raise ValueError(f"Tool package has no package path: {package_name}")

    registry = ToolRegistry()
    modules = sorted(
        pkgutil.iter_modules(package_path),
        key=lambda module_info: module_info.name,
    )
    for module_info in modules:
        module_name = f"{package_name}.{module_info.name}"
        module = importlib.import_module(module_name)
        factory = getattr(module, TOOL_FACTORY_NAME, None)
        if factory is None:
            continue
        if not callable(factory):
            raise TypeError(
                f"{module_name}.{TOOL_FACTORY_NAME} must be callable"
            )
        for definition in factory(context):
            if not isinstance(definition, ToolDefinition):
                raise TypeError(
                    f"{module_name}.{TOOL_FACTORY_NAME} returned "
                    f"{type(definition).__name__}, expected ToolDefinition"
                )
            registry.register(definition)
    return registry
