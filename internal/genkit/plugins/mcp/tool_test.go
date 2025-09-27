package mcp_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	internalmcp "github.com/habiliai/agentruntime/internal/genkit/plugins/mcp"
	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

func TestMCPToolCall(t *testing.T) {
	ctx := context.TODO()
	g := genkit.Init(ctx)

	c, err := mcpclient.NewStdioMCPClient("npx", []string{}, "-y", "@modelcontextprotocol/server-filesystem", ".")
	if err != nil {
		t.Fatalf("failed to create MCP client: %v", err)
	}
	defer c.Close()

	initRequest := mcp.InitializeRequest{}
	initRequest.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initRequest.Params.ClientInfo = mcp.Implementation{
		Name:    "example-client",
		Version: "1.0.0",
	}
	if _, err := c.Initialize(ctx, initRequest); err != nil {
		t.Fatalf("failed to initialize MCP client: %v", err)
	}

	listToolsRes, err := c.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		t.Fatalf("failed to list tools: %v", err)
	}
	var listDirTool mcp.Tool
	for _, tool := range listToolsRes.Tools {
		if tool.Name == "list_directory" {
			listDirTool = tool
			break
		}
	}
	tool, err := internalmcp.DefineTool(g, c, listDirTool, nil)
	if err != nil {
		t.Fatalf("failed to define tool: %v", err)
	}

	t.Run("Run Tool", func(t *testing.T) {
		out, err := tool.RunRaw(ctx, json.RawMessage(`{"path":"./"}`))
		if err != nil {
			t.Fatalf("failed to run tool: %v", err)
		}

		output, ok := out.(map[string]any)
		if !ok {
			t.Fatalf("result is not a ToolResult")
		}

		t.Logf("result: %+v", output)
	})
}
