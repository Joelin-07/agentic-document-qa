"""MCP server + client over real stdio (the server runs as a subprocess)."""

import pytest

from docqa.mcp_client import MCPDocumentClient, MCPToolError

from .conftest import DEMO_DOCS


@pytest.fixture
def mcp(settings):
    with MCPDocumentClient(settings) as client:
        yield client


def test_tools_are_listed_with_schemas(mcp):
    tools = {t["name"]: t for t in mcp.list_tools()}
    assert set(tools) == {"ingest_document", "list_documents", "search_documents", "read_document", "delete_document"}
    top_k = tools["search_documents"]["input_schema"]["properties"]["top_k"]
    assert top_k["maximum"] == 20 and top_k["default"] == 4


def test_ingest_search_read_delete_roundtrip(mcp):
    result = mcp.ingest(str(DEMO_DOCS / "security_policy.pdf"))
    assert result.status == "added" and result.document.file_type == "pdf"
    assert mcp.ingest(str(DEMO_DOCS / "security_policy.pdf")).status == "unchanged"

    [doc] = mcp.list_documents()
    hits = mcp.search("how often rotate API keys", top_k=2)
    assert hits.results[0].filename == "security_policy.pdf"
    assert "90 days" in mcp.read_document(doc.doc_id).text

    assert mcp.delete_document(doc.doc_id).deleted
    assert mcp.list_documents() == []


def test_pydantic_validation_errors_are_returned_as_tool_errors(mcp):
    with pytest.raises(MCPToolError, match="top_k"):
        mcp.search("x", top_k=99)
    with pytest.raises(MCPToolError, match="query"):
        mcp.call_tool("search_documents", {"query": ""})


def test_tool_exceptions_are_reported(mcp):
    with pytest.raises(MCPToolError, match="File not found"):
        mcp.ingest("does/not/exist.txt")
    with pytest.raises(MCPToolError, match="Unknown doc_id"):
        mcp.read_document("nope")
