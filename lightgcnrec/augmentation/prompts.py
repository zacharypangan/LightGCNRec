"""Prompt rendering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Mapping


@dataclass(frozen=True)
class PromptTemplate:
    """A named prompt template from the paper augmentation pipeline."""

    name: str
    entity_type: str
    purpose: str
    template: str


def render_prompt(prompt: PromptTemplate, values: Mapping[str, object]) -> str:
    """Render a prompt using `$field` placeholders."""

    return Template(prompt.template).safe_substitute({key: str(value) for key, value in values.items()})
