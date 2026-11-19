from client import MCPPermissionClient
import streamlit as st

st.set_page_config(page_title="MCP Security Lab", page_icon="🔐", layout="wide")

client = MCPPermissionClient()

if "pending_tool" not in st.session_state:
    st.session_state.pending_tool = None
if "pending_arguments" not in st.session_state:
    st.session_state.pending_arguments = None

def show_result(result):
    status = result.get("status")
    if status == "success":
        st.success(result["message"])
    elif status == "denied":
        st.error(result["message"])
    elif status == "approval_required":
        st.warning(result["message"])
        st.json(result)
    else:
        st.error(result.get("message", "Unknown error"))

st.title("MCP Security Lab")
st.write("Permission-aware MCP system using ALLOW / ASK / DENY policies.")

# Sidebar
st.sidebar.header("Security Policies")
tools = ["read_file", "write_file", "delete_file", "execute_command"]

for tool in tools:
    current = client.get_permission(tool)
    new_policy = st.sidebar.selectbox(
        tool,
        ["allow", "ask", "deny"],
        index=["allow", "ask", "deny"].index(current),
        key=f"permission_{tool}",
    )
    if new_policy != current:
        client.set_permission(tool, new_policy)

tab1, tab2, tab3 = st.tabs(["Tools", "Permissions", "Audit Log"])

# Tools Tab
with tab1:
    st.header("Execute MCP Tool")
    tool = st.selectbox("Select Tool", tools)
    st.info(f"Risk: {client.get_risk(tool).upper()} | Permission: {client.get_permission(tool).upper()}")

    if tool == "read_file":
        filepath = st.text_input("File path", "test.txt")
        if st.button("Call read_file"):
            result = client.call_tool("read_file", {"filepath": filepath})
            show_result(result)

    elif tool == "write_file":
        filepath = st.text_input("File path", "new_file.txt")
        content = st.text_area("Content", "Hello from MCP Security Lab")
        if st.button("Call write_file"):
            arguments = {"filepath": filepath, "content": content}
            result = client.call_tool("write_file", arguments)
            show_result(result)
            if result["status"] == "approval_required":
                st.session_state.pending_tool = "write_file"
                st.session_state.pending_arguments = arguments
                st.rerun()

        if st.session_state.pending_tool == "write_file":
            st.warning("This operation requires user approval.")
            st.json(st.session_state.pending_arguments)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve & Execute", key="approve_write"):
                    approved_result = client.call_tool("write_file", st.session_state.pending_arguments, approval=True)
                    st.session_state.pending_tool = None
                    st.session_state.pending_arguments = None
                    show_result(approved_result)
            with col2:
                if st.button("Reject", key="reject_write"):
                    client.log("REJECTED", "write_file", "User rejected approval request")
                    st.session_state.pending_tool = None
                    st.session_state.pending_arguments = None
                    st.warning("Operation rejected.")
                    st.rerun()

    elif tool == "delete_file":
        filepath = st.text_input("File path", "test.txt")
        if st.button("Call delete_file"):
            result = client.call_tool("delete_file", {"filepath": filepath})
            show_result(result)

    elif tool == "execute_command":
        command = st.text_input("Command", "echo Hello")
        if st.button("Call execute_command"):
            result = client.call_tool("execute_command", {"command": command})
            show_result(result)

# Permissions Tab
with tab2:
    st.header("Current Permissions")
    for tool in tools:
        permission = client.get_permission(tool)
        risk = client.get_risk(tool)
        st.write(f"**{tool}**")
        st.write(f"Permission: `{permission.upper()}`")
        st.write(f"Risk: `{risk.upper()}`")
        st.divider()

# Audit Log Tab
with tab3:
    st.header("Security Audit Log")
    st.code(client.get_audit_log())
    if st.button("Refresh Log"):
        st.rerun()
