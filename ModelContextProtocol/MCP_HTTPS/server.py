from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("HTTP File Server")

# Workspace root
BASE_DIR = Path(__file__).parent / "workspace"
BASE_DIR.mkdir(exist_ok=True)

# Security check
def is_within_workspace(filepath: Path) -> bool:
    """Check whether a path is inside workspace."""
    try:
        filepath.resolve().relative_to(BASE_DIR.resolve())
        return True
    except ValueError:
        return False

# Tool 1: Read File
@mcp.tool()
def read_file(filepath: str) -> str:
    """Read a file from the workspace."""
    path = BASE_DIR / filepath
    if not is_within_workspace(path):
        return "Error: Access denied - path outside workspace"
    if not path.exists():
        return f"Error: File not found: {filepath}"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

# Tool 2: Write File
@mcp.tool()
def write_file(filepath: str, content: str) -> str:
    """Write content to a file in the workspace."""
    path = BASE_DIR / filepath
    if not is_within_workspace(path):
        return "Error: Access denied - path outside workspace"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"

# Tool 3: List Files
@mcp.tool()
def list_files(directory: str = ".") -> str:
    """List files inside the workspace."""
    path = BASE_DIR / directory
    if not is_within_workspace(path):
        return "Error: Access denied - path outside workspace"
    if not path.exists():
        return f"Error: Directory not found: {directory}"
    if not path.is_dir():
        return f"Error: Not a directory: {directory}"

    try:
        files = []
        for item in sorted(path.iterdir()):
            relative_path = item.relative_to(BASE_DIR)
            if item.is_dir():
                file_type = "DIR"
                size = "-"
            else:
                file_type = "FILE"
                size = f"{item.stat().st_size} bytes"
            files.append(f"{file_type:5} {relative_path} {size}")
        return "\n".join(files) if files else "Directory is empty"
    except Exception as e:
        return f"Error listing directory: {e}"

# Tool 4: Analyze Code
@mcp.tool()
def analyze_code(code: str, focus: str = "quality") -> str:
    """Analyze code. Simplified demonstration of MCP Sampling."""
    return (
        "SAMPLING DEMONSTRATION\n\n"
        "In the full MCP architecture, the server would request an LLM completion "
        "from the client using sampling/createMessage.\n\n"
        f"Focus: {focus}\n\n"
        f"Code received:\n{code}"
    )

# MCP Resource
@mcp.resource("file:///workspace/{filename}")
def get_file_resource(filename: str) -> str:
    """Read a workspace file as an MCP resource."""
    path = BASE_DIR / filename
    if not is_within_workspace(path):
        return "Error: Access denied - path outside workspace"
    if not path.exists():
        return f"Error: File not found: {filename}"
    if not path.is_file():
        return f"Error: Not a file: {filename}"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading resource: {e}"

# MCP Prompts
@mcp.prompt()
def review_code(filename: str) -> str:
    """Generate a code review prompt for a file."""
    return f"""Review the code in the file '{filename}'.

Provide:
1. Summary of what the code does
2. Potential bugs
3. Code quality issues
4. Security concerns
5. Suggestions for improvement
6. Maintainability recommendations

Focus on readability, maintainability, correctness and best practices."""

@mcp.prompt()
def analyze_security(filename: str) -> str:
    """Generate a security analysis prompt."""
    return f"""Perform a security analysis of '{filename}'.

Focus on:
1. Input validation
2. Authentication and authorization
3. Injection vulnerabilities
4. Sensitive data exposure
5. Path traversal
6. Error handling
7. Other security risks

Provide specific recommendations for improving the code."""

# Start Server
if __name__ == "__main__":
    print("Starting MCP HTTP Server...")
    print(f"Workspace: {BASE_DIR}")
    print("Server: http://127.0.0.1:8000")
    mcp.run(transport="http", host="127.0.0.1", port=8000)
