import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
from mcp import Client, StdioServerParameters

BASE_DIR = Path(__file__).resolve().parent
SERVER_FILE = BASE_DIR / "server.py"
PERMISSIONS_FILE = BASE_DIR / "permissions.json"
AUDIT_FILE = BASE_DIR / "audit.log"

DEFAULT_PERMISSIONS = {
    "read_file": "allow",
    "write_file": "ask",
    "delete_file": "deny",
    "execute_command": "deny",
}

RISK_LEVELS = {
    "read_file": "low",
    "write_file": "medium",
    "delete_file": "high",
    "execute_command": "critical",
}

class MCPPermissionClient:
    def __init__(self):
        self.permissions = self.load_permissions()

    def load_permissions(self):
        if not PERMISSIONS_FILE.exists():
            self.save_permissions(DEFAULT_PERMISSIONS.copy())
            return DEFAULT_PERMISSIONS.copy()
        try:
            return json.loads(PERMISSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return DEFAULT_PERMISSIONS.copy()

    def save_permissions(self, permissions=None):
        if permissions is None:
            permissions = self.permissions
        PERMISSIONS_FILE.write_text(json.dumps(permissions, indent=4), encoding="utf-8")

    def set_permission(self, tool_name, policy):
        if policy not in ["allow", "ask", "deny"]:
            raise ValueError("Policy must be allow, ask, or deny.")
        self.permissions[tool_name] = policy
        self.save_permissions()
        self.log("PERMISSION_CHANGED", tool_name, policy)

    def get_permission(self, tool_name):
        return self.permissions.get(tool_name, "ask")

    def get_risk(self, tool_name):
        return RISK_LEVELS.get(tool_name, "medium")

    def log(self, event, tool_name, details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {event}: {tool_name}"
        if details:
            entry += f" | {details}"
        entry += "\n"
        with open(AUDIT_FILE, "a", encoding="utf-8") as file:
            file.write(entry)

    def get_audit_log(self):
        if not AUDIT_FILE.exists():
            return "No audit events yet."
        return AUDIT_FILE.read_text(encoding="utf-8")

    def get_server_parameters(self):
        return StdioServerParameters(command=sys.executable, args=[str(SERVER_FILE)])

    async def _call_mcp_tool(self, tool_name, arguments):
        server_params = self.get_server_parameters()
        async with Client(server_params) as client:
            return await client.call_tool(tool_name, arguments)

    async def _list_tools(self):
        server_params = self.get_server_parameters()
        async with Client(server_params) as client:
            result = await client.list_tools()
            return result.tools

    def list_tools(self):
        return asyncio.run(self._list_tools())

    def call_tool(self, tool_name, arguments, approval=False):
        permission = self.get_permission(tool_name)
        risk = self.get_risk(tool_name)

        if permission == "deny":
            self.log("DENIED", tool_name, f"risk={risk}, args={arguments}")
            return {
                "status": "denied",
                "message": f"Permission denied for tool: {tool_name}",
                "risk": risk,
                "permission": permission,
            }

        if permission == "ask" and not approval:
            self.log("ASK", tool_name, f"Awaiting approval, args={arguments}")
            return {
                "status": "approval_required",
                "message": f"Permission required for tool: {tool_name}",
                "risk": risk,
                "permission": permission,
                "arguments": arguments,
            }

        self.log("ALLOWED", tool_name, f"risk={risk}, args={arguments}")
        try:
            result = asyncio.run(self._call_mcp_tool(tool_name, arguments))
            text = self.extract_result(result)
            self.log("EXECUTED", tool_name, text)
            return {"status": "success", "message": text, "risk": risk, "permission": permission}
        except Exception as e:
            self.log("ERROR", tool_name, str(e))
            return {"status": "error", "message": f"MCP error: {str(e)}", "risk": risk, "permission": permission}

    @staticmethod
    def extract_result(result):
        if hasattr(result, "content"):
            outputs = [item.text if hasattr(item, "text") else str(item) for item in result.content]
            return "\n".join(outputs)
        return str(result)

if __name__ == "__main__":
    client = MCPPermissionClient()
    print("\n" + "=" * 50)
    print("MCP PERMISSION CLIENT")
    print("=" * 50)
    print(f"\nPython interpreter:\n{sys.executable}")
    print(f"\nMCP server:\n{SERVER_FILE}")
    print(f"\nPermissions:\n{json.dumps(client.permissions, indent=4)}")

    print("\nAvailable MCP tools:")
    try:
        tools = client.list_tools()
        for tool in tools:
            print(f"- {tool.name}")
    except Exception as e:
        print(f"Could not list tools: {e}")

    print("\nTesting read_file...")
    result = client.call_tool("read_file", {"filepath": "test.txt"})
    print(json.dumps(result, indent=4))
