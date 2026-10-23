"""Async MCP stdio client for a knowledge-palace server.

The MCP stdio transport context managers must be entered and exited in the
same task, so the connection lives in a dedicated background task and call
sites talk to it through readiness state + a lock.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..config import KnowledgeConfig

log = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the mcp package
    MCP_AVAILABLE = False


class KnowledgeBaseUnavailable(RuntimeError):
    pass


def _extract(result: Any) -> Any:
    """Flatten a CallToolResult into JSON-ish data or plain text."""
    texts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            texts.append(text)
    joined = "\n".join(texts).strip()
    if getattr(result, "isError", False):
        raise KnowledgeBaseUnavailable(f"knowledge-palace tool error: {joined[:300]}")
    if not joined:
        return None
    try:
        return json.loads(joined)
    except (json.JSONDecodeError, ValueError):
        return joined


class McpStdioConnection:
    def __init__(self, command: list[str], env: dict[str, str] | None = None):
        if not MCP_AVAILABLE:
            raise KnowledgeBaseUnavailable(
                "the 'mcp' package is not installed (pip install mcp)"
            )
        if not command:
            raise KnowledgeBaseUnavailable("knowledge.command is empty")
        self._command = command
        self._env = env or {}
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self._session: Any = None
        self._task: asyncio.Task | None = None
        self._error: Exception | None = None
        self._tools: list[str] = []

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self._error = None
        self._task = asyncio.create_task(self._run(), name="kp-mcp-connection")

    async def _run(self) -> None:
        params = StdioServerParameters(command=self._command[0], args=self._command[1:], env=self._env or None)
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=30)
                    async with self._lock:
                        self._session = session
                        try:
                            listed = await session.list_tools()
                            self._tools = [t.name for t in listed.tools]
                        except Exception:
                            self._tools = []
                    log.info("knowledge-palace connected, tools: %s", self._tools)
                    self._ready.set()
                    await self._stop.wait()
        except Exception as exc:
            self._error = exc
            log.warning("knowledge-palace connection failed: %s", exc)
        finally:
            async with self._lock:
                self._session = None
            self._ready.set()  # wake any waiters; they will see the closed session

    async def call(self, tool: str, arguments: dict[str, Any], timeout: float = 30.0) -> Any:
        await asyncio.wait_for(self._ready.wait(), timeout=60)
        session = self._session
        if session is None:
            raise KnowledgeBaseUnavailable(
                f"knowledge-palace is not connected: {self._error or 'server not reachable'}"
            )
        async with self._lock:
            result = await asyncio.wait_for(session.call_tool(tool, arguments), timeout=timeout)
        return _extract(result)

    async def list_tools(self) -> list[str]:
        await asyncio.wait_for(self._ready.wait(), timeout=60)
        if self._session is None:
            raise KnowledgeBaseUnavailable(
                f"knowledge-palace is not connected: {self._error or 'server not reachable'}"
            )
        return list(self._tools)

    def is_alive(self) -> bool:
        return self._session is not None

    def needs_restart(self) -> bool:
        """True when the connection task exited or failed (not while handshaking)."""
        return self._task is None or self._task.done() or self._error is not None

    def error(self) -> str | None:
        return str(self._error) if self._error else None

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        self._session = None


class KnowledgeService:
    """Typed facade over the knowledge-palace MCP server."""

    def __init__(self, config: KnowledgeConfig):
        self.config = config
        self._conn: McpStdioConnection | None = None

    async def start(self) -> None:
        if not self.config.enabled:
            return
        try:
            self._conn = McpStdioConnection(self.config.command, self.config.env)
            await self._conn.start()
        except KnowledgeBaseUnavailable as exc:
            log.warning("knowledge base disabled: %s", exc)
            self._conn = None

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def available(self) -> bool:
        return self._conn is not None

    def _require(self) -> McpStdioConnection:
        if self._conn is None:
            raise KnowledgeBaseUnavailable("knowledge base is disabled or failed to start")
        return self._conn

    async def _ensure_alive(self) -> McpStdioConnection:
        """Reconnect transparently if the previous session died."""
        conn = self._require()
        if conn.needs_restart():
            await self.close()
            await self.start()
        return self._require()

    async def tools(self) -> list[str]:
        return await self._require().list_tools()

    async def status(self) -> dict:
        if not self.config.enabled or self._conn is None:
            return {"enabled": self.config.enabled, "connected": False,
                    "command": self.config.command}
        if self._conn.needs_restart():
            return {"enabled": True, "connected": False,
                    "command": self.config.command,
                    "error": self._conn.error() or "server exited"}
        try:
            tools = await asyncio.wait_for(self.tools(), timeout=30)
            return {"enabled": True, "connected": True,
                    "command": self.config.command, "tools": tools}
        except Exception as exc:
            return {"enabled": True, "connected": False,
                    "command": self.config.command, "error": str(exc)}

    async def search(self, query: str, limit: int | None = None,
                     mode: str | None = None) -> Any:
        args: dict[str, Any] = {"query": query, "limit": limit or self.config.default_top_k}
        if mode:
            args["mode"] = mode
        conn = await self._ensure_alive()
        return await conn.call("search", args, self.config.timeout_s)

    async def ask(self, question: str, limit: int | None = None) -> Any:
        conn = await self._ensure_alive()
        return await conn.call(
            "ask", {"question": question, "limit": limit or self.config.default_top_k},
            self.config.timeout_s * 2,
        )

    async def context(self, query: str, limit: int | None = None) -> Any:
        conn = await self._ensure_alive()
        return await conn.call(
            "context", {"query": query, "limit": limit or self.config.default_top_k},
            self.config.timeout_s,
        )

    async def similar(self, text: str, limit: int = 5) -> Any:
        conn = await self._ensure_alive()
        return await conn.call(
            "similar", {"text": text, "limit": limit}, self.config.timeout_s
        )
