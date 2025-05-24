package openaiapi

import (
	"encoding/json"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/habiliai/agentruntime/internal/genkit/plugins/internal/config"
	goopenai "github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
)

func convertRequest(model string, input *ai.ModelRequest) (goopenai.ChatCompletionNewParams, error) {
	messages, err := convertMessages(input.Messages)
	if err != nil {
		return goopenai.ChatCompletionNewParams{}, err
	}

	tools, err := convertTools(input.Tools)
	if err != nil {
		return goopenai.ChatCompletionNewParams{}, err
	}

	chatCompletionRequest := goopenai.ChatCompletionNewParams{
		Model:    goopenai.String(model),
		Messages: goopenai.F(messages),
	}

	if len(tools) > 0 {
		chatCompletionRequest.Tools = goopenai.F(tools)
	}

	jsonBytes, err := json.Marshal(input.Config)
	if err != nil {
		return goopenai.ChatCompletionNewParams{}, err
	}
	{
		var c ai.GenerationCommonConfig
		if err := json.Unmarshal(jsonBytes, &c); err == nil {
			if c.MaxOutputTokens != 0 {
				chatCompletionRequest.MaxTokens = goopenai.Int(int64(c.MaxOutputTokens))
			}
			if len(c.StopSequences) > 0 {
				chatCompletionRequest.Stop = goopenai.F[goopenai.ChatCompletionNewParamsStopUnion](goopenai.ChatCompletionNewParamsStopArray(c.StopSequences))
			}
			if c.Temperature != 0 {
				chatCompletionRequest.Temperature = goopenai.Float(c.Temperature)
			}
			if c.TopP != 0 {
				chatCompletionRequest.TopP = goopenai.Float(c.TopP)
			}
		}
	}
	{
		var c config.GenerationReasoningConfig
		if err := json.Unmarshal(jsonBytes, &c); err == nil {
			if c.ReasoningEffort != "" {
				chatCompletionRequest.ReasoningEffort = goopenai.F(goopenai.ChatCompletionReasoningEffort(c.ReasoningEffort))
			}
		}
	}

	if input.Output != nil &&
		input.Output.Format != "" {
		switch input.Output.Format {
		case ai.OutputFormatJSON:
			chatCompletionRequest.ResponseFormat = goopenai.F[goopenai.ChatCompletionNewParamsResponseFormatUnion](
				goopenai.ChatCompletionNewParamsResponseFormat{
					Type: goopenai.F(goopenai.ChatCompletionNewParamsResponseFormatTypeJSONObject),
				},
			)
		case ai.OutputFormatText:
			chatCompletionRequest.ResponseFormat = goopenai.F[goopenai.ChatCompletionNewParamsResponseFormatUnion](
				goopenai.ChatCompletionNewParamsResponseFormat{
					Type: goopenai.F(goopenai.ChatCompletionNewParamsResponseFormatTypeText),
				},
			)
		default:
			return goopenai.ChatCompletionNewParams{}, fmt.Errorf("unknown output format in a request: %s", input.Output.Format)
		}
	}

	return chatCompletionRequest, nil
}

func convertMessages(messages []*ai.Message) ([]goopenai.ChatCompletionMessageParamUnion, error) {
	var msgs []goopenai.ChatCompletionMessageParamUnion

	for _, m := range messages {
		switch m.Role {
		case ai.RoleSystem: // system
			var text string
			for _, content := range m.Content {
				if content.Text != "" {
					text += content.Text
				}
			}
			sm := goopenai.SystemMessage(text)
			msgs = append(msgs, sm)
		case ai.RoleUser: // user
			var multiContent []goopenai.ChatCompletionContentPartUnionParam
			for _, p := range m.Content {
				part, err := convertPart(p)
				if err != nil {
					return nil, err
				}
				multiContent = append(multiContent, part)
			}
			um := goopenai.UserMessageParts(multiContent...)
			msgs = append(msgs, um)
		case ai.RoleModel: // assistant
			toolCalls, err := convertToolCalls(m.Content)
			if err != nil {
				return nil, err
			}
			am := goopenai.ChatCompletionAssistantMessageParam{
				Role: goopenai.F(goopenai.ChatCompletionAssistantMessageParamRoleAssistant),
			}
			if m.Content[0].Text != "" {
				am.Content = goopenai.F([]goopenai.ChatCompletionAssistantMessageParamContentUnion{
					goopenai.TextPart(m.Content[0].Text),
				})
			}
			if len(toolCalls) > 0 {
				am.ToolCalls = goopenai.F(toolCalls)
			}
			msgs = append(msgs, am)
		case ai.RoleTool: // tool
			for _, p := range m.Content {
				if !p.IsToolResponse() {
					continue
				}
				output, err := json.Marshal(p.ToolResponse.Output)
				if err != nil {
					return nil, err
				}
				tm := goopenai.ToolMessage(
					p.ToolResponse.Name,
					string(output),
				)
				msgs = append(msgs, tm)
			}
		default:
			return nil, fmt.Errorf("Unknown OpenAI Role %s", m.Role)
		}
	}

	return msgs, nil
}

func convertPart(part *ai.Part) (res goopenai.ChatCompletionContentPartUnionParam, err error) {
	switch {
	case part.IsText():
		res = goopenai.TextPart(part.Text)
	case part.IsMedia():
		res = goopenai.ChatCompletionContentPartImageParam{
			Type: goopenai.F(goopenai.ChatCompletionContentPartImageTypeImageURL),
			ImageURL: goopenai.F(goopenai.ChatCompletionContentPartImageImageURLParam{
				URL:    goopenai.F(part.Text),
				Detail: goopenai.F(goopenai.ChatCompletionContentPartImageImageURLDetailAuto),
			}),
		}
	default:
		err = fmt.Errorf("unknown part type in a request: %#v", part)
	}
	return
}

func convertToolCalls(content []*ai.Part) ([]goopenai.ChatCompletionMessageToolCallParam, error) {
	var toolCalls []goopenai.ChatCompletionMessageToolCallParam
	for _, p := range content {
		if !p.IsToolRequest() {
			continue
		}
		toolCall, err := convertToolCall(p)
		if err != nil {
			return nil, err
		}
		toolCalls = append(toolCalls, toolCall)
	}
	return toolCalls, nil
}

func convertToolCall(part *ai.Part) (goopenai.ChatCompletionMessageToolCallParam, error) {
	param := goopenai.ChatCompletionMessageToolCallParam{
		ID:   goopenai.F(part.ToolRequest.Name),
		Type: goopenai.F(goopenai.ChatCompletionMessageToolCallTypeFunction),
		Function: goopenai.F(goopenai.ChatCompletionMessageToolCallFunctionParam{
			Name: goopenai.F(part.ToolRequest.Name),
		}),
	}

	if part.ToolRequest.Input != nil {
		var args []byte
		args, err := json.Marshal(part.ToolRequest.Input)
		if err != nil {
			return param, err
		}
		param.Function.Value.Arguments = goopenai.F(string(args))
	} else {
		// NOTE: OpenAI API requires the Arguments field to be set even if it's empty.
		param.Function.Value.Arguments = goopenai.F("{}")
	}

	return param, nil
}

func convertTools(inTools []*ai.ToolDefinition) ([]goopenai.ChatCompletionToolParam, error) {
	var tools []goopenai.ChatCompletionToolParam
	for _, t := range inTools {
		tool, err := convertTool(t)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

func convertTool(t *ai.ToolDefinition) (goopenai.ChatCompletionToolParam, error) {
	return goopenai.ChatCompletionToolParam{
		Type: goopenai.F(goopenai.ChatCompletionToolTypeFunction),
		Function: goopenai.F(shared.FunctionDefinitionParam{
			Name:        goopenai.F(t.Name),
			Description: goopenai.F(t.Description),
			Parameters:  goopenai.F(goopenai.FunctionParameters(t.InputSchema)),
			Strict:      goopenai.F(false),
		}),
	}, nil
}
