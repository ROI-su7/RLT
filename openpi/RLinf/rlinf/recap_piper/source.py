"""Read task/model settings without importing or changing the native checkout."""

from __future__ import annotations

import ast
from pathlib import Path


def native_config(path: Path, name: str) -> tuple[str, dict]:
    """Extract literal task and Pi0Config kwargs from a local TrainConfig.

    Refuse expressions rather than silently substituting a different task.
    Parsing instead of executing the file avoids JAX/GPU initialization.
    """
    tree = ast.parse(Path(path).read_text())
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        kwargs = {item.arg: item.value for item in call.keywords if item.arg}
        try:
            matches = "name" in kwargs and ast.literal_eval(kwargs["name"]) == name
        except (ValueError, TypeError):
            matches = False
        if not matches:
            continue
        data, model = kwargs.get("data"), kwargs.get("model")
        if not isinstance(data, ast.Call) or not isinstance(model, ast.Call):
            raise ValueError(f"{name}: expected literal data/model constructors")
        data_kwargs = {item.arg: item.value for item in data.keywords if item.arg}
        prompt = ast.literal_eval(data_kwargs["default_prompt"])
        architecture = {
            item.arg: ast.literal_eval(item.value)
            for item in model.keywords
            if item.arg
        }
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not architecture.get("pi05")
        ):
            raise ValueError(
                f"{name}: expected a pi05 model and explicit non-empty prompt"
            )
        return prompt, architecture
    raise ValueError(f"Native OpenPI config {name!r} not found in {path}")
