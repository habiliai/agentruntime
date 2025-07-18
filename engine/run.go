package engine

import (
	"context"
	_ "embed"
	"encoding/json"
	"math"
	"text/template"

	"github.com/firebase/genkit/go/ai"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/tool"
	"github.com/pkg/errors"
)

const (
	defaultMaxTurns = math.MaxInt
)

var (
	//go:embed data/instructions/chat.md.tmpl
	chatInst     string
	chatInstTmpl *template.Template = template.Must(template.New("").Funcs(funcMap()).Parse(chatInst))
)

type (
	Action struct {
		Name      string `json:"name"`
		Arguments any    `json:"arguments"`
		Result    any    `json:"result"`
	}
	Conversation struct {
		User    string   `json:"user,omitempty"`
		Text    string   `json:"text,omitempty"`
		Actions []Action `json:"actions,omitempty"`
	}

	AvailableAction struct {
		Action      string `json:"action"`
		Description string `json:"description"`
	}

	Participant struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Role        string `json:"role"`
	}

	Thread struct {
		Instruction  string
		Participants []Participant `json:"participants,omitempty"`
	}

	ChatPromptValues struct {
		Agent               entity.Agent
		RecentConversations []Conversation
		AvailableActions    []AvailableAction
		MessageExamples     [][]entity.MessageExample
		Thread              Thread
		Tools               []ai.ToolRef
		System              string
	}

	RunRequest struct {
		ThreadInstruction string         `json:"thread_instruction,omitempty"`
		History           []Conversation `json:"history"`
		Participant       []Participant  `json:"participants,omitempty"`
	}

	RunResponse struct {
		*ai.ModelResponse
		ToolCalls []ToolCall `json:"tool_calls"`
	}

	ToolCall struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		Result    json.RawMessage `json:"result"`
	}
)

func (s *Engine) Run(
	ctx context.Context,
	agent entity.Agent,
	req RunRequest,
	streamCallback ai.ModelStreamCallback,
) (*RunResponse, error) {

	promptValues, err := s.BuildPromptValues(ctx, agent, req.History, Thread{
		Instruction:  req.ThreadInstruction,
		Participants: req.Participant,
	})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to build prompt values")
	}

	ctx = tool.WithEmptyCallDataStore(ctx)
	res := &RunResponse{}
	for i := 0; i < 3; i++ {
		res.ModelResponse, err = s.Generate(
			ctx,
			&GenerateRequest{
				Model: agent.ModelName,
			},
			ai.WithSystem(promptValues.System),
			ai.WithPromptFn(GetPromptFn(promptValues)),
			ai.WithConfig(agent.ModelConfig),
			ai.WithTools(promptValues.Tools...),
			ai.WithStreaming(streamCallback),
			ai.WithMaxTurns(defaultMaxTurns),
		)
		if err != nil {
			s.logger.Warn("failed to generate", "err", err)
		} else {
			break
		}
	}
	if err != nil {
		return nil, errors.Wrapf(err, "failed to generate")
	}

	toolCallData := tool.GetCallData(ctx)
	for _, data := range toolCallData {
		tc := ToolCall{
			Name: data.Name,
		}

		if v, err := json.Marshal(data.Arguments); err != nil {
			return nil, errors.Wrapf(err, "failed to marshal tool call arguments")
		} else {
			tc.Arguments = v
		}

		if v, err := json.Marshal(data.Result); err != nil {
			return nil, errors.Wrapf(err, "failed to marshal tool call result")
		} else {
			tc.Result = v
		}

		res.ToolCalls = append(res.ToolCalls, tc)
	}

	return res, nil
}
