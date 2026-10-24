"""MCP stdio stub that mimics knowledge-palace's tool set, for tests.

Used by tests/test_knowledge_guard.py. Supports both the mcp 2.x and 1.x
server APIs; mirrors the real KP behavior of answering unknown tools with
plain text ("Unknown tool: x").

    python stub_kp_server.py [--with-context]
"""

import asyncio
import json
import sys

WITH_CONTEXT = "--with-context" in sys.argv

CONTEXT_HITS = json.dumps(
    {"results": [
        {"text": "PEP hits above 70% fuzzy match go to the EDD team.",
         "source": "AML policy", "score": 0.91},
    ]}
)


def _run_2x() -> None:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        "kp-stub",
        instructions="stub knowledge-palace: search/ask/similar"
        + (", context" if WITH_CONTEXT else ""),
    )

    @server.tool(description="Search the knowledge base.")
    async def search(query: str, limit: int = 5) -> str:
        return "Found 1 results (hybrid search):\n--- Result 1 ---\nPEP hits require EDD."

    @server.tool(description="Ask the knowledge base.")
    async def ask(question: str, limit: int = 5) -> str:
        return "Based on your knowledge base:\n[1] PEP hits go to the EDD team."

    @server.tool(description="Find similar content.")
    async def similar(text: str, limit: int = 5) -> str:
        return "Found 1 similar passages."

    if WITH_CONTEXT:
        @server.tool(description="Structured hits for decision grounding.")
        async def context(query: str, limit: int = 5) -> str:
            return CONTEXT_HITS

    server.run(transport="stdio")


async def _run_1x() -> None:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    tools = [
        Tool(name="search", description="search", inputSchema=dict(schema)),
        Tool(name="ask", description="ask", inputSchema=dict(schema)),
        Tool(name="similar", description="similar", inputSchema=dict(schema)),
    ]
    if WITH_CONTEXT:
        tools.append(Tool(name="context", description="grounding hits", inputSchema=dict(schema)))

    async def handle(name: str, arguments: dict) -> list:
        if name == "context":
            return [TextContent(type="text", text=CONTEXT_HITS)]
        if name == "search":
            return [TextContent(type="text", text="Found 1 results:\nPEP hits require EDD.")]
        if name == "ask":
            return [TextContent(type="text", text="[1] PEP hits go to the EDD team.")]
        if name == "similar":
            return [TextContent(type="text", text="1 similar passage.")]
        # faithful to the real KP server: plain text, not an MCP error
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    server = Server("kp-stub")

    @server.list_tools()
    async def list_tools():
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        return await handle(name, arguments)

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    try:
        import mcp.server.mcpserver  # noqa: F401

        has_2x = True
    except ImportError:
        has_2x = False
    if has_2x:
        _run_2x()
    else:
        asyncio.run(_run_1x())


if __name__ == "__main__":
    main()
