package tool

import (
	"context"
	"log/slog"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/mylog"
	"github.com/habiliai/agentruntime/knowledge"
	"github.com/habiliai/agentruntime/memory"
	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/pkg/errors"
)

type (
	Manager interface {
		GetTool(toolName string) ai.Tool
		GetMCPTool(serverName, toolName string) ai.Tool
		GetMCPTools(ctx context.Context, serverName string) []ai.Tool
		GetToolsBySkill(ctx context.Context, skill entity.AgentSkillUnion) ([]ai.Tool, error)
		GetUsagePrompt(skill entity.AgentSkillUnion) string
		Close()
	}
	manager struct {
		logger *mylog.Logger

		mcpClients     map[string]*mcpclient.Client
		mtx            sync.Mutex
		genkit         *genkit.Genkit
		skillToolNames map[string][]string // skill.Name -> tool names
		usagePrompts   map[string]string

		knowledgeService knowledge.Service
		memoryService    memory.Service
	}
)

func (m *manager) GetTool(toolName string) ai.Tool {
	return genkit.LookupTool(m.genkit, toolName)
}

var (
	_ Manager = (*manager)(nil)
)

func NewToolManager(ctx context.Context, skills []entity.AgentSkillUnion, logger *slog.Logger, genkit *genkit.Genkit, knowledgeService knowledge.Service, memoryService memory.Service) (Manager, error) {
	s := &manager{
		logger:           logger,
		mcpClients:       make(map[string]*mcpclient.Client),
		genkit:           genkit,
		knowledgeService: knowledgeService,
		memoryService:    memoryService,
		skillToolNames:   make(map[string][]string),
		usagePrompts:     make(map[string]string),
	}

	for _, skill := range skills {
		switch skill.Type {
		case "mcp":
			if err := s.registerMCPSkill(ctx, skill.OfMCP); err != nil {
				return nil, errors.Wrapf(err, "failed to register mcp skill")
			}
		case "llm":
			if err := s.registerLLMSkill(ctx, skill.OfLLM); err != nil {
				return nil, errors.Wrapf(err, "failed to register llm skill")
			}
		case "nativeTool":
			if err := s.registerNativeSkill(skill.OfNative); err != nil {
				return nil, errors.Wrapf(err, "failed to register native skill")
			}
		default:
			return nil, errors.Errorf("invalid skill type: %s", skill.Type)
		}
	}

	return s, nil
}

func (m *manager) GetMCPTool(serverName, toolName string) ai.Tool {
	if _, ok := m.mcpClients[serverName]; !ok {
		return nil
	}

	return genkit.LookupTool(m.genkit, toolName)
}

func (m *manager) GetUsagePrompt(skill entity.AgentSkillUnion) string {
	skillName := ""
	switch skill.Type {
	case "nativeTool":
		skillName = skill.OfNative.Name
	case "llm":
		skillName = skill.OfLLM.Name
	case "mcp":
		skillName = skill.OfMCP.Name
	}

	if _, ok := m.usagePrompts[skillName]; !ok {
		return ""
	}

	return m.usagePrompts[skillName]
}

func (m *manager) Close() {
	for _, client := range m.mcpClients {
		if err := client.Close(); err != nil {
			return
		}
	}
}

func (m *manager) GetToolsBySkill(ctx context.Context, skill entity.AgentSkillUnion) ([]ai.Tool, error) {
	switch skill.Type {
	case "llm", "nativeTool":
		skillName := func() string {
			switch skill.Type {
			case "llm":
				return skill.OfLLM.Name
			case "nativeTool":
				return skill.OfNative.Name
			default:
				return ""
			}
		}()

		toolNames, ok := m.skillToolNames[skillName]
		if !ok || len(toolNames) == 0 {
			return nil, errors.Errorf("no tools found for skill %s", skillName)
		}
		tools := make([]ai.Tool, 0, len(toolNames))
		for _, toolName := range toolNames {
			tools = append(tools, m.GetTool(toolName))
		}
		return tools, nil
	case "mcp":
		skillToolNames := skill.OfMCP.Tools
		if len(skillToolNames) == 0 {
			return m.GetMCPTools(ctx, skill.OfMCP.Name), nil
		}
		tools := make([]ai.Tool, 0, len(skillToolNames))
		for _, skillToolName := range skillToolNames {
			tool := m.GetMCPTool(skill.OfMCP.Name, skillToolName)
			if tool == nil {
				return nil, errors.Errorf("invalid tool name %s", skill.OfMCP.Name)
			}
			tools = append(tools, tool)
		}
		return tools, nil
	}

	return nil, errors.Errorf("invalid skill type: %s", skill.Type)
}
