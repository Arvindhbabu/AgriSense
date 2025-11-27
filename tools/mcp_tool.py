# tools/mcp_tool.py
"""
Lightweight MCP client wrapper. The ADK notebooks used an `npx @modelcontextprotocol/server-everything` server.
This class expects either:
 - gateway_url set to a local HTTP gateway for your MCP server, OR
 - ability to call the ADK MCP python object directly (if used within the same runtime).
"""
import os
import requests
import logging

logger = logging.getLogger("agrisense.mcp")

class MCPToolClient:
    def __init__(self, gateway_url: str = None):
        self.gateway = gateway_url or os.environ.get("MCP_GATEWAY_URL")  # e.g., http://localhost:8080

    def call(self, tool_name: str, payload: dict, timeout: int = 15):
        if self.gateway:
            url = f"{self.gateway}/call_tool"
            try:
                resp = requests.post(url, json={"tool": tool_name, "payload": payload}, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.error("MCP call failed: %s", e)
                raise
        else:
            raise RuntimeError("No MCP gateway configured. Set MCP_GATEWAY_URL env var or pass a client object.")
