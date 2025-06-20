package tool

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	internalmcp "github.com/habiliai/agentruntime/internal/genkit/plugins/mcp"
	mcpclient "github.com/mark3labs/mcp-go/client"
	mcp "github.com/mark3labs/mcp-go/mcp"
	"github.com/pkg/errors"
)

type RegisterMCPToolRequest struct {
	ServerID string
	Command  string
	Args     []string
	Env      map[string]string
}

func (m *manager) registerMCPTool(ctx context.Context, req RegisterMCPToolRequest) (err error) {
	m.mtx.Lock()
	defer m.mtx.Unlock()

	var envs []string
	for key, val := range req.Env {
		envs = append(envs, fmt.Sprintf("%s=%s", key, val))
	}

	mcpClient, ok := m.mcpClients[req.ServerID]
	if !ok {
		c, err := mcpclient.NewStdioMCPClient(req.Command, envs, req.Args...)
		if err != nil {
			return fmt.Errorf("failed to create MCP client: %w", err)
		}

		stderr, ok := mcpclient.GetStderr(c)
		if ok {
			go func(stderr io.Reader) {
				rd := bufio.NewReader(stderr)
				for {
					line, err := rd.ReadString('\n')
					if err != nil {
						if err == io.EOF || strings.Contains(err.Error(), "already closed") {
							return
						}
						m.logger.Error("failed to copy stderr", "err", err, "serverName", req.ServerID)
						return
					}
					m.logger.Warn("[MCP] "+strings.TrimSpace(line), "serverName", req.ServerID)
				}
			}(stderr)
		}

		initRequest := mcp.InitializeRequest{}
		initRequest.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
		if _, err := c.Initialize(ctx, initRequest); err != nil {
			return errors.Wrapf(err, "failed to initialize MCP client")
		}

		m.mcpClients[req.ServerID] = c
		mcpClient = c
	}

	listToolsResult, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return errors.Wrapf(err, "failed to list tools")
	}
	for _, tool := range listToolsResult.Tools {
		if genkit.LookupTool(m.genkit, tool.Name) != nil {
			m.logger.InfoContext(ctx, "tool already registered", "tool", tool.Name)
			continue
		}
		if _, err := internalmcp.DefineTool(m.genkit, mcpClient, tool, func(ctx *ai.ToolContext, in any, out *internalmcp.ToolResult) error {
			appendCallData(ctx, CallData{
				Name:      tool.Name,
				Arguments: in,
				Result:    out,
			})
			return nil
		}); err != nil {
			return errors.Wrapf(err, "failed to define tool")
		}
	}

	return nil
}

func (m *manager) GetMCPTools(ctx context.Context, mcpServerName string) []ai.Tool {
	client, ok := m.mcpClients[mcpServerName]
	if !ok {
		return nil
	}

	listToolsResult, err := client.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		m.logger.Error("failed to list tools", "err", err)
		return nil
	}

	var tools []ai.Tool
	for _, tool := range listToolsResult.Tools {
		if t := genkit.LookupTool(m.genkit, tool.Name); t != nil {
			tools = append(tools, t)
		}
	}

	return tools
}
