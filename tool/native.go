package tool

import (
	"github.com/habiliai/agentruntime/entity"
	"github.com/pkg/errors"
)

func registerNativeTool[In any, Out any](m *manager, toolName, toolDescription string, skill *entity.NativeAgentSkill, fn func(ctx *Context, input In) (Out, error)) error {
	toolNames := m.skillToolNames[skill.Name]

	for _, existingToolName := range toolNames {
		if existingToolName == toolName {
			return errors.Errorf("tool %s already registered", toolName)
		}
	}

	registerLocalTool(m, toolName, toolDescription, skill, fn)
	m.skillToolNames[skill.Name] = append(toolNames, toolName)

	return nil
}

func (m *manager) registerNativeSkill(skill *entity.NativeAgentSkill) error {
	if skill.Name == "" {
		return errors.New("native tool name is required")
	}
	switch skill.Name {
	case "get_weather":
		return m.registerGetWeatherTool(skill)
	case "knowledge_search":
		return m.registerKnowledgeSearchTool(skill)
	case "rss":
		return m.registerRSSSkill(skill)
	case "memory":
		return m.registerMemorySKill(skill)
	}

	return nil
}
