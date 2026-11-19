from pathlib import Path
from mcp.server import MCPServer

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

mcp = MCPServer("Secure MCP Server")

def get_safe_path(filepath: str) -> Path:
    """Convert a user-provided path into a safe path inside the data directory."""
    path = (DATA_DIR / filepath).resolve()
    if not str(path).startswith(str(DATA_DIR.resolve())):
        raise ValueError("Access outside data directory is not allowed.")
    return path

# Low Risk Tool
@mcp.tool()
def read_file(filepath: str) -> str:
    """Read a file from the data directory. Risk: LOW"""
    try:
        file_path = get_safe_path(filepath)
        if not file_path.exists():
            return f"Error: File '{filepath}' not found."
        if not file_path.is_file():
            return f"Error: '{filepath}' is not a file."
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

# Medium Risk Tool
@mcp.tool()
def write_file(filepath: str, content: str) -> str:
    """Write content to a file in the data directory. Risk: MEDIUM"""
    try:
        file_path = get_safe_path(filepath)
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"

# High Risk Tool
@mcp.tool()
def delete_file(filepath: str) -> str:
    """Delete a file from the data directory. Risk: HIGH"""
    try:
        file_path = get_safe_path(filepath)
        if not file_path.exists():
            return f"Error: File '{filepath}' not found."
        file_path.unlink()
        return f"Successfully deleted {filepath}"
    except Exception as e:
        return f"Error deleting file: {e}"

# Critical Risk Tool
@mcp.tool()
def execute_command(command: str) -> str:
    """Simulated system command execution. Risk: CRITICAL"""
    return (
        "SIMULATED COMMAND EXECUTION\n\n"
        f"Command: {command}\n\n"
        "No actual system command was executed."
    )

if __name__ == "__main__":
    mcp.run()
