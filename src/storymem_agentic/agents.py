from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any, Protocol


class AgentBackend(Protocol):
    def generate_json(self, prompt: str, schema: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class MockAgentBackend:
    name: str = "mock"
    responses: dict[str, dict[str, Any]] = field(default_factory=dict)

    def generate_json(self, prompt: str, schema: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        response_key = str(context.get("response_key", "default"))
        if response_key in self.responses:
            return self.responses[response_key]
        return {"backend": self.name, "prompt": prompt[:500], "schema_name": schema.get("name", "json"), "context": context}


@dataclass(slots=True)
class CommandAgentBackend:
    command_template: str
    call_count: int = 0

    def generate_json(self, prompt: str, schema: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        payload = json.dumps({"prompt": prompt, "schema": schema, "context": context})
        command = shlex.split(self.command_template)
        try:
            result = subprocess.run(command, input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"agent backend failed with exit code {exc.returncode}: {exc.stderr or exc.stdout}"
            ) from exc
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"agent backend did not return JSON: {result.stdout[:500]}") from exc
