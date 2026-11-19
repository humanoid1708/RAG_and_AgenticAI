from datetime import datetime
from pathlib import Path
from fastmcp import Context, FastMCP
from pydantic import BaseModel

BASE_DIR = Path.cwd().resolve()

class DocumentGeneratorSchema(BaseModel):
    """Input schema used by the documentation prompt."""
    file_path: str
    name: str

mcp = FastMCP("File Operations MCP Server")

def get_path(relative_path: str) -> Path:
    """Convert a relative path to a safe absolute path."""
    path = (BASE_DIR / relative_path).resolve()
    try:
        path.relative_to(BASE_DIR)
    except ValueError:
        raise ValueError("Path is outside the project directory.")
    return path

# Tools
@mcp.tool()
async def write_file(file_path: str, content: str, ctx: Context) -> str:
    """Create a file with the specified content."""
    try:
        path = get_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        total = len(content)
        chunk_size = max(total // 10, 1)
        written = 0

        with open(path, "w", encoding="utf-8") as f:
            for i in range(0, total, chunk_size):
                chunk = content[i : i + chunk_size]
                f.write(chunk)
                written += len(chunk)
                await ctx.report_progress(progress=written, total=total, message="Writing file...")

        await ctx.info(f"File written successfully: {file_path}")
        return f"File written successfully to: {file_path}"
    except Exception as e:
        await ctx.error(f"Error creating file: {str(e)}")
        raise

@mcp.tool()
async def delete_file(file_path: str, ctx: Context) -> str:
    """Delete a file from the project directory."""
    try:
        path = get_path(file_path)
        if path.is_file():
            path.unlink()
            await ctx.info(f"Successfully deleted file: {file_path}")
            return f"Successfully deleted file: {file_path}"
        if path.is_dir():
            await ctx.warning(f"Error: {file_path} is a directory, not a file.")
            return f"Error: {file_path} is a directory, not a file."
        await ctx.warning(f"File not found: {file_path}")
        return f"File not found: {file_path}"
    except Exception as e:
        await ctx.error(f"Error deleting file: {str(e)}")
        return f"Error deleting file: {str(e)}"

# Resources
@mcp.resource("file:///{file_name}")
async def read_file_resource(file_name: str) -> dict:
    """Read a file through an MCP resource."""
    try:
        path = get_path(file_name)
        if not path.exists() or not path.is_file():
            return {"error": f"{file_name} is not a valid file"}
        return {"file_content": path.read_text(encoding="utf-8")}
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}

@mcp.resource("dir:///")
async def list_files_resource() -> dict:
    """List files and directories in the project directory."""
    try:
        path = get_path(".")
        items = []
        for item in path.iterdir():
            stat = item.stat()
            items.append({
                "name": item.name,
                "path": str(item.relative_to(BASE_DIR)),
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            })
        return {"items": items}
    except Exception as e:
        return {"error": f"Error listing files: {str(e)}"}

# Prompts
@mcp.prompt()
async def code_review(file_path: str, ctx: Context) -> str:
    """Generate a code-review prompt for a file."""
    try:
        path = get_path(file_path)
        if not path.exists() or not path.is_file():
            error_msg = f"Error: {file_path} is not a valid file"
            await ctx.warning(error_msg)
            raise FileNotFoundError(error_msg)

        current_code = path.read_text(encoding="utf-8").strip()
        language = path.suffix.lower()

        prompt = f"""You are an expert code editor.

Review the following code comprehensively.

File:
{file_path}

Language:
{language if language else "unknown"}

Current code:
----------------
{current_code}
----------------

Provide a comprehensive evaluation of:
1. Bugs
2. Code quality
3. Performance
4. Security
5. Maintainability
6. Suggested improvements""".strip()

        await ctx.info("Successfully generated code review prompt.")
        return prompt
    except Exception as e:
        await ctx.error(f"Error preparing code review prompt: {e}")
        raise

@mcp.prompt()
async def documentation_generator(ctx: Context) -> str:
    """Generate a prompt for creating documentation."""
    try:
        result = await ctx.elicit(
            message="Please provide the source file path and documentation filename.",
            response_type=DocumentGeneratorSchema,
        )

        if result.action != "accept":
            return "Documentation generation was cancelled."

        file_path = get_path(result.data.file_path)
        doc_name = result.data.name

        if not file_path.exists() or not file_path.is_file():
            error_msg = f"Error: {result.data.file_path} is not a valid file"
            await ctx.warning(error_msg)
            raise FileNotFoundError(error_msg)

        code = file_path.read_text(encoding="utf-8").strip()
        language = file_path.suffix.lower()

        prompt = f"""You are an expert technical writer.

Create comprehensive documentation for:

File:
{result.data.file_path}

Language:
{language if language else "unknown"}

Current code:
----------------
{code}
----------------

Create a separate documentation file named:
{doc_name}

The documentation should explain:
1. Purpose
2. Architecture
3. Functions
4. Parameters
5. Return values
6. Usage
7. Important implementation details

Use the MCP tools available to create the documentation file.

IMPORTANT:
The documentation filename must be exactly:
{doc_name}""".strip()

        await ctx.info("Successfully generated documentation prompt.")
        return prompt
    except Exception as e:
        await ctx.error(f"Error generating documentation prompt: {e}")
        raise

if __name__ == "__main__":
    print("Starting File Operations MCP Server...")
    mcp.run()
