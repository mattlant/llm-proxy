#!/usr/bin/env python3
"""Offline, installed-wheel proof for the deterministic mock provider.

Run from the repository root with ``--wheelhouse .temp/mat27-wheelhouse``.
The child process runs from a temporary directory, never from this checkout.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


CHILD = r'''
import asyncio
import json
import os
import socket
import tempfile
from pathlib import Path

import httpx

config = """server:
  host: 127.0.0.1
  port: 11435
  timeout_seconds: 1
  config_reload_seconds: 60
  log_level: INFO
management:
  enabled: true
  token_env: MAT27_ADMIN_TOKEN
  command_timeout_seconds: 1
interfaces:
  openai: {enabled: true}
  ollama: {enabled: true}
  anthropic:
    enabled: true
    authentication: {allow_any_token: true, token: null}
providers:
  first:
    extension: deterministic-mock
    enabled: true
    config: {scenario: {version: 1, rules: [{id: reply, responses: [{text: installed}]}]}}
  second:
    extension: deterministic-mock
    enabled: true
    config: {scenario: {version: 1, rules: [{id: reply, responses: [{text: second}]}]}}
  builtin:
    extension: ollama
    enabled: true
    config: {base_url: http://127.0.0.1:UPSTREAM_PORT, outbound_interface: openai}
models:
  mock-model:
    upstream_model: mock-upstream
    provider: first
    interfaces: [openai, ollama, anthropic]
    parameters: {}
  builtin-model:
    upstream_model: builtin-upstream
    provider: builtin
    interfaces: [openai]
    parameters: {}
policies: []
"""
reserve = socket.socket()
reserve.bind(("127.0.0.1", 0))
upstream_port = reserve.getsockname()[1]
reserve.close()
config = config.replace("UPSTREAM_PORT", str(upstream_port))
directory = tempfile.TemporaryDirectory(prefix="mat27-config-")
path = Path(directory.name) / "config.yaml"
path.write_text(config)
os.environ["LLM_PROXY_CONFIG"] = str(path)
os.environ["MAT27_ADMIN_TOKEN"] = "mat27-token"

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore

app = create_app(ConfigurationStore(path, ConfigurationLoader(), 60))

async def main():
    async def upstream(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        payload = json.dumps({"id": "builtin-response", "model": "builtin-upstream", "choices": [{"index": 0, "message": {"role": "assistant", "content": "built-in"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(payload)).encode() + b"\r\nConnection: close\r\n\r\n" + payload)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(upstream, "127.0.0.1", upstream_port)
    headers = {"Authorization": "Bearer mat27-token"}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
            models = await client.get("/v1/models")
            assert {item["id"]: item["owned_by"] for item in models.json()["data"]}["builtin-model"] == "builtin"
            built_in = await client.post("/v1/chat/completions", json={"model": "builtin-model", "messages": [{"role": "user", "content": "hello"}]})
            openai = await client.post("/v1/chat/completions", json={"model": "mock-model", "messages": [{"role": "user", "content": "hello"}]})
            ollama = await client.post("/api/generate", json={"model": "mock-model", "prompt": "hello", "stream": False})
            anthropic = await client.post("/v1/messages", headers={"x-api-key": "any", "anthropic-version": "2023-06-01"}, json={"model": "mock-model", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}]})
            assert built_in.json()["choices"][0]["message"]["content"] == "built-in"
            assert openai.json()["choices"][0]["message"]["content"] == "installed"
            assert ollama.json()["response"] == "installed"
            assert anthropic.json()["content"] == [{"type": "text", "text": "installed"}]
            commands = await client.get("/_admin/v1/providers/first/commands", headers=headers)
            assert {item["name"] for item in commands.json()["commands"]} == {"replace_scenario", "enqueue_response", "inspect_captured_requests", "inspect_pending_expectations", "reset_state", "verify_expectations"}
            payloads = {
                "replace_scenario": {"scenario": {"version": 1, "rules": [{"id": "reply", "responses": [{"text": "installed"}]}]}},
                "enqueue_response": {"response": {"text": "queued"}},
                "inspect_captured_requests": {}, "inspect_pending_expectations": {}, "reset_state": {}, "verify_expectations": {},
            }
            for name, payload in payloads.items():
                response = await client.post(f"/_admin/v1/providers/first/commands/{name}", headers=headers, json=payload)
                assert response.status_code == 200, response.text
    finally:
        server.close()
        await server.wait_closed()

asyncio.run(main())
print("installed gateway interface and admin proof passed")
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if not args.wheelhouse.is_dir():
        raise SystemExit(f"wheelhouse does not exist: {args.wheelhouse}")
    with tempfile.TemporaryDirectory(prefix="mat27-wheels-") as wheels, tempfile.TemporaryDirectory(prefix="mat27-venv-") as venv:
        subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", wheels], cwd=root, check=True)
        subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", wheels], cwd=root / "extensions" / "mock_provider", check=True)
        subprocess.run([sys.executable, "-m", "venv", venv], check=True)
        python = Path(venv) / "bin" / "python"
        subprocess.run([python, "-m", "pip", "install", "--no-index", "--find-links", args.wheelhouse, "--find-links", wheels, "llm-proxy==0.4.0", "llm-proxy-mock-provider==0.1.0"], check=True)
        subprocess.run([python, "-c", CHILD], cwd="/tmp", check=True)


if __name__ == "__main__":
    main()
