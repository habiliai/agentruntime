package tool

import (
	"context"
	"slices"

	"github.com/gosimple/slug"
	"github.com/habiliai/agentruntime/entity"
	"github.com/pkg/errors"
)

type (
	LLMToolRequest  struct{}
	LLMToolResponse struct {
		Instruction string `json:"additional_important_instruction" jsonschema:"description=Additional important instruction to the LLM"`
	}
)

func (m *manager) registerLLMTool(_ context.Context, name, description, instruction string) {
	registerLocalTool(
		m,
		name,
		description,
		nil,
		func(ctx *Context, req struct {
			*LLMToolRequest
		}) (res struct {
			*LLMToolResponse
		}, err error) {
			res.LLMToolResponse = &LLMToolResponse{
				Instruction: instruction,
			}
			return
		},
	)
}

func (m *manager) registerLLMSkill(ctx context.Context, skill *entity.LLMAgentSkill) error {
	if skill.Name == "" {
		return errors.New("llm name is required")
	}
	if skill.Description == "" {
		return errors.New("llm description is required")
	}
	if skill.Instruction == "" {
		return errors.New("llm instruction is required")
	}
	toolName := slug.Make(skill.Name)
	if !CanBeUsedAsToolName(toolName) {
		return errors.New("llm tool name is not valid. only accept by [a-zA-Z0-9_-]{1,128}")
	}
	m.registerLLMTool(ctx, toolName, skill.Description, skill.Instruction)

	if _, ok := m.skillToolNames[skill.Name]; ok {
		if slices.Contains(m.skillToolNames[skill.Name], toolName) {
			return errors.New("llm tool name already registered")
		}
	}
	m.skillToolNames[skill.Name] = append(m.skillToolNames[skill.Name], toolName)

	return nil
}
