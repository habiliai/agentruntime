package tool

import (
	"context"
	"log/slog"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/mylog"
	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/pkg/errors"
)

type (
	Manager interface {
		GetTool(toolName string) ai.Tool
		GetMCPTool(serverName, toolName string) ai.Tool
		GetMCPTools(ctx context.Context, serverName string) []ai.Tool
		Close()
	}
	manager struct {
		logger *mylog.Logger

		mcpClients map[string]mcpclient.MCPClient
		mtx        sync.Mutex
		genkit     *genkit.Genkit
	}
)

var (
	_ Manager = (*manager)(nil)
)

func NewToolManager(ctx context.Context, skills []entity.AgentSkill, logger *slog.Logger, genkit *genkit.Genkit) (Manager, error) {
	s := &manager{
		logger:     logger,
		mcpClients: make(map[string]mcpclient.MCPClient),
		genkit:     genkit,
	}

	for _, skill := range skills {
		switch skill.Type {
		case "mcp":
			if skill.Name == "" {
				return nil, errors.New("mcp server is required")
			}
			if skill.Command == "" {
				return nil, errors.New("mcp command is required")
			}
			if err := s.registerMCPTool(ctx, RegisterMCPToolRequest{
				ServerID: skill.Name,
				Command:  skill.Command,
				Args:     skill.Args,
				Env:      skill.Env,
			}); err != nil {
				return nil, errors.Wrap(err, "failed to register mcp tool")
			}
		case "llm":
			if skill.Name == "" {
				return nil, errors.New("llm name is required")
			}
			if skill.Description == "" {
				return nil, errors.New("llm description is required")
			}
			if skill.Instruction == "" {
				return nil, errors.New("llm instruction is required")
			}
			s.registerLLMTool(ctx, skill.Name, skill.Description, skill.Instruction)
		case "nativeTool":
			if skill.Name == "" {
				return nil, errors.New("native tool name is required")
			}
			s.registerNativeTool(skill.Name, skill.Description, skill.Env)
		default:
			return nil, errors.Errorf("invalid skill type: %s", skill.Type)
		}
	}

	return s, nil
}

func (m *manager) GetTool(toolName string) ai.Tool {
	return genkit.LookupTool(m.genkit, toolName)
}

func (m *manager) GetMCPTool(serverName, toolName string) ai.Tool {
	if _, ok := m.mcpClients[serverName]; !ok {
		return nil
	}

	return genkit.LookupTool(m.genkit, toolName)
}

func (m *manager) Close() {
	for _, client := range m.mcpClients {
		if err := client.Close(); err != nil {
			return
		}
	}
}

func (m *manager) registerNativeTool(name string, description string, env map[string]string) {
	switch strings.ToLower(name) {
	case "get_weather":
		m.registerGetWeatherTool(description, env)
	}
}

func registerLocalTool[In any, Out any](m *manager, name string, description string, fn func(context.Context, In) (Out, error)) ai.Tool {
	tool := m.GetTool(name)
	if tool != nil {
		return tool
	}

	return genkit.DefineTool(
		m.genkit,
		name,
		description,
		func(ctx *ai.ToolContext, input In) (Out, error) {
			out, err := fn(ctx, input)
			if err == nil {
				appendCallData(ctx, CallData{
					Name:      name,
					Arguments: input,
					Result:    out,
				})
			}
			return out, err
		},
	)
}
