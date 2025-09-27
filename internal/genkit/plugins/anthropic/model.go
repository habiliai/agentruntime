package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/habiliai/agentruntime/internal/version"
	"github.com/pkg/errors"
)

func HttpGet(url string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create request")
	}

	req.Header.Set("User-Agent", "AgentRuntime/"+version.Version)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, "failed to send request")
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errors.Errorf("failed to get URL: %s", resp.Status)
	}
	return resp, nil
}

// DefineModel creates and registers a new generative model with Genkit.
func (a *Anthropic) DefineModel(labelPrefix, provider, modelName, apiModelName string, caps ai.ModelSupports) ai.Model {
	return ai.NewModel(
		api.NewName(provider, modelName),
		&ai.ModelOptions{
			Label:    labelPrefix + " - " + modelName,
			Supports: &caps,
		},
		func(ctx context.Context, req *ai.ModelRequest, cb core.StreamCallback[*ai.ModelResponseChunk]) (*ai.ModelResponse, error) {
			if cb == nil {
				// Non-streaming generation
				return a.generate(ctx, req, apiModelName)
			}
			// Streaming generation
			return a.generateStream(ctx, req, apiModelName, cb)
		},
	)
}

func (a *Anthropic) generate(ctx context.Context, genRequest *ai.ModelRequest, apiModelName string) (*ai.ModelResponse, error) {
	params, err := buildMessageParams(genRequest, apiModelName, false)
	if err != nil {
		return nil, err
	}

	// Use standard Messages API (which supports extended thinking for Claude 4 models)
	resp, err := a.client.Beta.Messages.New(ctx, params)
	if err != nil {
		return nil, errors.Wrapf(err, "anthropic message generation failed")
	}

	return translateResponse(*resp, genRequest)
}

func (a *Anthropic) generateStream(ctx context.Context, genRequest *ai.ModelRequest, apiModelName string, cb core.StreamCallback[*ai.ModelResponseChunk]) (*ai.ModelResponse, error) {
	params, err := buildMessageParams(genRequest, apiModelName, false)
	if err != nil {
		return nil, err
	}

	// Use standard streaming API
	stream := a.client.Beta.Messages.NewStreaming(ctx, params)
	defer stream.Close()

	type ToolUsePart struct {
		Ref     string
		Name    string
		Input   string
		Enabled bool
	}
	type WebSearchToolResultPart struct {
		Ref     string
		Name    string
		Result  string
		Enabled bool
	}
	var toolUsePart ToolUsePart
	var webSearchToolResultPart WebSearchToolResultPart

	message := anthropic.BetaMessage{}
	for stream.Next() {
		event := stream.Current()
		err := message.Accumulate(event)
		if err != nil {
			return nil, errors.Wrapf(err, "error accumulating message")
		}
		switch event := event.AsAny().(type) {
		case anthropic.BetaRawContentBlockStartEvent:
			switch block := event.ContentBlock.AsAny().(type) {
			case anthropic.BetaToolUseBlock:
				toolUsePart = ToolUsePart{
					Ref:     block.ID,
					Name:    block.Name,
					Input:   "",
					Enabled: true,
				}
			case anthropic.BetaServerToolUseBlock:
				toolUsePart = ToolUsePart{
					Ref:     block.ID,
					Name:    string(block.Name),
					Input:   "",
					Enabled: true,
				}
			case anthropic.BetaWebSearchToolResultBlock:
				webSearchToolResultPart = WebSearchToolResultPart{
					Ref:     block.ToolUseID,
					Name:    "web_search",
					Result:  block.Content.RawJSON(),
					Enabled: true,
				}
			}
		case anthropic.BetaRawContentBlockDeltaEvent:
			chunk := &ai.ModelResponseChunk{
				Index:      int(event.Index),
				Role:       ai.RoleModel,
				Aggregated: false,
			}

			switch delta := event.Delta.AsAny().(type) {
			case anthropic.BetaTextDelta:
				chunk.Content = []*ai.Part{ai.NewTextPart(delta.Text)}
				if err := cb(ctx, chunk); err != nil {
					return nil, err
				}
			case anthropic.BetaInputJSONDelta:
				if !toolUsePart.Enabled {
					return nil, errors.New("received input JSON delta but no tool use part found")
				}
				toolUsePart.Input += delta.PartialJSON
			case anthropic.BetaThinkingDelta:
				chunk.Content = []*ai.Part{ai.NewReasoningPart(delta.Thinking, []byte{})}
				if err := cb(ctx, chunk); err != nil {
					return nil, err
				}
			case anthropic.BetaSignatureDelta:
				chunk.Content = []*ai.Part{ai.NewReasoningPart("", []byte(delta.Signature))}
				if err := cb(ctx, chunk); err != nil {
					return nil, err
				}
			case anthropic.BetaCitationsDelta:
				var citation map[string]any
				if err := json.Unmarshal([]byte(delta.Citation.RawJSON()), &citation); err != nil {
					return nil, errors.Wrapf(err, "could not unmarshal citation delta into citation type")
				}
				chunk.Content = []*ai.Part{ai.NewCustomPart(map[string]any{
					"type": "citation",
					"body": citation,
				})}
				if err := cb(ctx, chunk); err != nil {
					return nil, err
				}
			}
		case anthropic.BetaRawContentBlockStopEvent:
			if toolUsePart.Enabled {
				toolUsePart.Enabled = false
				chunk := &ai.ModelResponseChunk{
					Index:      int(event.Index),
					Role:       ai.RoleModel,
					Aggregated: false,
					Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
						Ref:   toolUsePart.Ref,
						Name:  toolUsePart.Name,
						Input: json.RawMessage(toolUsePart.Input),
					})},
				}
				if err := cb(ctx, chunk); err != nil {
					return nil, err
				}
			}
			if webSearchToolResultPart.Enabled {
				webSearchToolResultPart.Enabled = false
				chunk := &ai.ModelResponseChunk{
					Index:      int(event.Index),
					Role:       ai.RoleModel,
					Aggregated: false,
					Content: []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{
						Ref:    webSearchToolResultPart.Ref,
						Name:   webSearchToolResultPart.Name,
						Output: json.RawMessage(webSearchToolResultPart.Result),
					})},
				}
				if err := cb(ctx, chunk); err != nil {
					return nil, err
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		return nil, errors.Wrapf(err, "anthropic streaming error")
	}

	return translateResponse(message, genRequest)
}

func buildMessageParams(genRequest *ai.ModelRequest, apiModelName string, downloadUrl bool) (anthropic.BetaMessageNewParams, error) {
	messages, systems, err := convertMessages(genRequest.Messages, genRequest.Docs, downloadUrl)
	if err != nil {
		return anthropic.BetaMessageNewParams{}, err
	}

	defaultParams, ok := defaultModelParams[apiModelName]
	if !ok {
		return anthropic.BetaMessageNewParams{}, errors.Errorf("model %s not found", apiModelName)
	}

	params := anthropic.BetaMessageNewParams{
		Model:    anthropic.Model(apiModelName),
		Messages: messages,
		Betas: []anthropic.AnthropicBeta{
			anthropic.AnthropicBetaContext1m2025_08_07,
			anthropic.AnthropicBetaInterleavedThinking2025_05_14,
		},
		System: systems,
	}

	if genRequest.Config == nil {
		genRequest.Config = map[string]any{}
	}

	// Handle generation config
	jsonBytes, err := json.Marshal(genRequest.Config)
	if err != nil {
		return anthropic.BetaMessageNewParams{}, err
	}

	// Extract and apply extended thinking config
	type configWithExtendedThinking struct {
		ai.GenerationCommonConfig
		ExtendedThinkingConfig
	}

	// Start with defaults
	config := configWithExtendedThinking{
		GenerationCommonConfig: defaultParams.GenerationCommonConfig,
		ExtendedThinkingConfig: defaultParams.ExtendedThinkingConfig,
	}

	// Unmarshal user config - this will only override provided fields
	if err := json.Unmarshal(jsonBytes, &config); err != nil {
		return anthropic.BetaMessageNewParams{}, errors.Wrapf(err, "failed to unmarshal config")
	}

	// Apply basic config
	if config.MaxOutputTokens > 0 {
		params.MaxTokens = int64(config.MaxOutputTokens)
	} else {
		return anthropic.BetaMessageNewParams{}, errors.New("maxOutputTokens is required")
	}
	if config.Temperature > 0 {
		params.Temperature = anthropic.Float(config.Temperature)
	}
	if config.TopP > 0 {
		params.TopP = anthropic.Float(config.TopP)
	}
	if config.TopK > 0 {
		params.TopK = anthropic.Int(int64(config.TopK))
	}
	if len(config.StopSequences) > 0 {
		params.StopSequences = config.StopSequences
	}

	// Apply extended thinking configuration
	if config.ExtendedThinkingEnabled {
		// Calculate budget based on ratio
		budgetRatio := config.ExtendedThinkingBudgetRatio
		if budgetRatio == 0 {
			budgetRatio = 0.15 // Default to 15% if not specified
		}

		budget := int64(float64(config.MaxOutputTokens) * budgetRatio)

		// Only enable if budget meets minimum requirement (1024 tokens)
		if budget >= 1024 {
			params.Thinking = anthropic.BetaThinkingConfigParamOfEnabled(budget)
		}
	}

	// Handle tools if present
	if len(genRequest.Tools) > 0 {
		tools := make([]anthropic.BetaToolUnionParam, len(genRequest.Tools))
		for i, tool := range genRequest.Tools {
			switch tool.Name {
			case "web_search":
				tools[i] = anthropic.BetaToolUnionParam{
					OfWebSearchTool20250305: &anthropic.BetaWebSearchTool20250305Param{
						MaxUses: anthropic.Int(99),
					},
				}
			default:
				tools[i] = convertTool(tool)
			}
		}
		params.Tools = tools
	}

	return params, nil
}

func convertMessages(messages []*ai.Message, docs []*ai.Document, downloadUrl bool) ([]anthropic.BetaMessageParam, []anthropic.BetaTextBlockParam, error) {
	var systems []anthropic.BetaTextBlockParam
	var anthropicMessages []anthropic.BetaMessageParam

	for _, doc := range docs {
		blocks, err := convertContent(doc.Content, downloadUrl)
		if err != nil {
			return nil, nil, err
		}
		anthropicMessages = append(anthropicMessages, anthropic.BetaMessageParam{
			Role:    anthropic.BetaMessageParamRoleUser,
			Content: blocks,
		})
	}

	for _, msg := range messages {
		var role anthropic.BetaMessageParamRole
		switch msg.Role {
		case ai.RoleUser:
			role = anthropic.BetaMessageParamRoleUser
		case ai.RoleModel:
			role = anthropic.BetaMessageParamRoleAssistant
		case ai.RoleTool:
			role = anthropic.BetaMessageParamRoleUser
		case ai.RoleSystem:
			for _, part := range msg.Content {
				if part.IsText() && part.Text != "" {
					text := strings.TrimSpace(part.Text)
					if text == "" {
						continue
					}
					systems = append(systems, anthropic.BetaTextBlockParam{
						Text: text,
					})
				}
			}
			continue
		default:
			return nil, nil, errors.Errorf("unsupported message role: %s", msg.Role)
		}

		content, err := convertContent(msg.Content, downloadUrl)
		if err != nil {
			return nil, nil, err
		}

		anthropicMessages = append(anthropicMessages, anthropic.BetaMessageParam{
			Role:    role,
			Content: content,
		})
	}

	return anthropicMessages, systems, nil
}

func convertContent(parts []*ai.Part, downloadUrl bool) ([]anthropic.BetaContentBlockParamUnion, error) {
	var blocks []anthropic.BetaContentBlockParamUnion

	for _, part := range parts {
		if part.IsCustom() {
			custom := part.Custom
			customType, ok := custom["type"].(string)
			if !ok {
				return nil, errors.New("custom type not found in custom part")
			}
			body := custom["body"]
			switch customType {
			case "web_search_tool_result":
				block, ok := body.(anthropic.BetaWebSearchToolResultBlock)
				if !ok {
					return nil, errors.New("body is not a web search tool result block")
				}
				blocks = append(blocks, anthropic.BetaContentBlockParamUnion{
					OfWebSearchToolResult: &anthropic.BetaWebSearchToolResultBlockParam{
						Type:      block.Type,
						ToolUseID: block.ToolUseID,
						Content: anthropic.BetaWebSearchToolResultBlockParamContentUnion{
							OfError: func() *anthropic.BetaWebSearchToolRequestErrorParam {
								err := block.Content.AsResponseWebSearchToolResultError()
								if err.Type == "" || err.ErrorCode == "" {
									return nil
								}
								return &anthropic.BetaWebSearchToolRequestErrorParam{
									ErrorCode: err.ErrorCode,
									Type:      err.Type,
								}
							}(),
							OfResultBlock: func() (v []anthropic.BetaWebSearchResultBlockParam) {
								arr := block.Content.OfBetaWebSearchResultBlockArray
								if len(arr) == 0 {
									return nil
								}

								for _, item := range arr {
									v = append(v, anthropic.BetaWebSearchResultBlockParam{
										Type:             item.Type,
										URL:              item.URL,
										Title:            item.Title,
										EncryptedContent: item.EncryptedContent,
										PageAge:          anthropic.String(item.PageAge),
									})
								}

								return v
							}(),
						},
					},
				})
			case "redacted_thinking":
				block, ok := body.(anthropic.BetaRedactedThinkingBlock)
				if !ok {
					return nil, errors.New("body is not a redacted thinking block")
				}
				blocks = append(blocks, anthropic.BetaContentBlockParamUnion{
					OfRedactedThinking: &anthropic.BetaRedactedThinkingBlockParam{
						Type: block.Type,
						Data: block.Data,
					},
				})
			case "server_tool_use":
				block, ok := body.(anthropic.BetaServerToolUseBlock)
				if !ok {
					return nil, errors.New("body is not a server tool use block")
				}
				blocks = append(blocks, anthropic.BetaContentBlockParamUnion{
					OfServerToolUse: &anthropic.BetaServerToolUseBlockParam{
						ID:    block.ID,
						Input: block.Input,
						Name:  anthropic.BetaServerToolUseBlockParamName(block.Name),
						Type:  block.Type,
					},
				})
			default:
				return nil, errors.Errorf("unsupported custom type: %s", customType)
			}
		} else if part.IsReasoning() {
			signature, ok := part.Metadata["signature"].([]byte)
			if !ok {
				return nil, errors.New("signature not found in reasoning part")
			}
			blocks = append(blocks, anthropic.NewBetaThinkingBlock(string(signature), part.Text))
		} else if part.IsText() {
			// Use the NewTextBlock helper function
			blocks = append(blocks, anthropic.NewBetaTextBlock(part.Text))
		} else if part.IsMedia() {
			// Handle image content
			data := part.Text

			isHttpsUrl := strings.HasPrefix(data, "https://")
			if strings.HasPrefix(data, "http://") || (downloadUrl && isHttpsUrl) {
				resp, err := HttpGet(data)
				if err != nil {
					return nil, errors.Wrapf(err, "failed to get URL")
				}
				defer resp.Body.Close()
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					return nil, errors.Wrapf(err, "failed to read URL body")
				}

				if part.ContentType == "plain/text" || part.ContentType == "text/plain" {
					data = string(body)
				} else {
					data = base64.StdEncoding.EncodeToString(body)
				}
			} else if strings.HasPrefix(data, "data:") {
				if !strings.Contains(data, ";base64,") {
					return nil, errors.New("data URL is not base64 encoded")
				}
				parts := strings.SplitN(data, ",", 2)
				if len(parts) == 2 {
					data = parts[1]
				}
			}

			switch strings.ToLower(part.ContentType) {
			case "image/jpeg", "image/png", "image/webp", "image/gif", "image/jpg":
				// Create image block with base64 source
				if isHttpsUrl && !downloadUrl {
					blocks = append(blocks, anthropic.NewBetaImageBlock(anthropic.BetaURLImageSourceParam{
						URL: data,
					}))
				} else {
					blocks = append(blocks, anthropic.NewBetaImageBlock(anthropic.BetaBase64ImageSourceParam{
						Data:      data,
						MediaType: getAnthropicMediaType(part.ContentType),
					}))
				}
			case "application/pdf":
				if isHttpsUrl && !downloadUrl {
					blocks = append(blocks, anthropic.NewBetaDocumentBlock(anthropic.BetaURLPDFSourceParam{
						URL: data,
					}))
				} else {
					blocks = append(blocks, anthropic.NewBetaDocumentBlock(anthropic.BetaBase64PDFSourceParam{
						Data: data,
					}))
				}
			case "plain/text", "text/plain":
				if isHttpsUrl && !downloadUrl {
					resp, err := HttpGet(data)
					if err != nil {
						return nil, errors.Wrapf(err, "failed to get URL")
					}
					defer resp.Body.Close()
					body, err := io.ReadAll(resp.Body)
					if err != nil {
						return nil, errors.Wrapf(err, "failed to read URL body")
					}

					data = string(body)
				}
				blocks = append(blocks, anthropic.NewBetaDocumentBlock(anthropic.BetaPlainTextSourceParam{
					Data: data,
				}))
			default:
				return nil, errors.Errorf("unsupported media type: %s", part.ContentType)
			}

		} else if part.IsToolRequest() {
			// Convert tool request to Anthropic format
			toolReq := part.ToolRequest

			// Marshal the input to get the string representation
			var inputMsg json.RawMessage
			if toolReq.Input != nil {
				inputJSON, err := json.Marshal(toolReq.Input)
				if err != nil {
					return nil, errors.Wrapf(err, "failed to marshal tool input")
				}
				inputMsg = inputJSON
			}
			if len(inputMsg) == 0 {
				inputMsg = json.RawMessage("{}")
			}

			toolUse := anthropic.NewBetaToolUseBlock(
				toolReq.Ref,  // ID
				inputMsg,     // Input as JSON string
				toolReq.Name, // Name
			)
			blocks = append(blocks, toolUse)
		} else if part.IsToolResponse() {
			// Convert tool response to Anthropic format
			toolResp := part.ToolResponse

			contents, err := convertToolResultBlockContents(toolResp.Output)
			if err != nil {
				resultBlock := anthropic.NewBetaToolResultBlock(
					toolResp.Ref,
				)
				resultBlock.OfToolResult.Content = []anthropic.BetaToolResultBlockParamContentUnion{
					{
						OfText: &anthropic.BetaTextBlockParam{
							Text: err.Error(),
						},
					},
				}
				resultBlock.OfToolResult.IsError = anthropic.Opt(true)
				blocks = append(blocks, resultBlock)
			} else {
				blocks = append(blocks, anthropic.BetaContentBlockParamUnion{
					OfToolResult: &anthropic.BetaToolResultBlockParam{
						ToolUseID: toolResp.Ref,
						Content:   contents,
						IsError:   anthropic.Opt(false),
					},
				})
			}
		}
	}

	return blocks, nil
}

func convertToolResultBlockContents(output any) (contents []anthropic.BetaToolResultBlockParamContentUnion, err error) {
	// Handle tool response content
	switch v := output.(type) {
	case string:
		contents = append(contents, anthropic.BetaToolResultBlockParamContentUnion{
			OfText: &anthropic.BetaTextBlockParam{
				Text: v,
			},
		})
	case []any:
		for _, item := range v {
			children, err := convertToolResultBlockContents(item)
			if err != nil {
				return nil, err
			}
			contents = append(contents, children...)
		}
	case map[string]any:
		if v, ok := v["error"]; ok {
			var content string
			switch err := v.(type) {
			case string:
				content = err
			case error:
				content = err.Error()
			default:
				content = fmt.Sprintf("%v", err)
			}

			if content != "" {
				contents = append(contents, anthropic.BetaToolResultBlockParamContentUnion{
					OfText: &anthropic.BetaTextBlockParam{
						Text: content,
					},
				})
			}
		}
		if output, ok := v["output"]; ok {
			children, err := convertToolResultBlockContents(output)
			if err != nil {
				return nil, err
			}
			contents = append(contents, children...)
			break
		}

		if contentType, ok := v["contentType"].(string); ok {
			if url, ok := v["url"].(string); ok {
				// Handle ai.Media type

				var source anthropic.BetaImageBlockParamSourceUnion
				if strings.HasPrefix(url, "http://") {
					resp, err := http.Get(url)
					if err != nil {
						return nil, err
					}
					defer resp.Body.Close()
					body, err := io.ReadAll(resp.Body)
					if err != nil {
						return nil, err
					}
					source.OfBase64 = &anthropic.BetaBase64ImageSourceParam{
						Data:      base64.StdEncoding.EncodeToString(body),
						MediaType: getAnthropicMediaType(contentType),
					}
				} else if strings.HasPrefix(url, "https://") {
					source.OfURL = &anthropic.BetaURLImageSourceParam{
						URL: url,
					}
				} else {
					source.OfBase64 = &anthropic.BetaBase64ImageSourceParam{
						Data:      url,
						MediaType: getAnthropicMediaType(contentType),
					}
				}

				contents = append(contents, anthropic.BetaToolResultBlockParamContentUnion{
					OfImage: &anthropic.BetaImageBlockParam{
						Source: source,
					},
				})
				break
			}
		}

		// Marshal non-string outputs to JSON
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}

		contents = append(contents, anthropic.BetaToolResultBlockParamContentUnion{
			OfText: &anthropic.BetaTextBlockParam{
				Text: string(jsonBytes),
			},
		})

	default:
		return nil, errors.Errorf("unsupported tool result block content type: %T", v)
	}

	return
}

func convertTool(tool *ai.ToolDefinition) anthropic.BetaToolUnionParam {
	// Convert the InputSchema map to ToolInputSchemaParam
	inputSchema := anthropic.BetaToolInputSchemaParam{
		Type:       "object",
		Properties: tool.InputSchema["properties"],
	}

	return anthropic.BetaToolUnionParam{
		OfTool: &anthropic.BetaToolParam{
			Name:        tool.Name,
			Description: anthropic.String(tool.Description),
			InputSchema: inputSchema,
		},
	}
}

func translateContent(content anthropic.BetaContentBlockUnion) *ai.Part {
	switch content.Type {
	case "text":
		return ai.NewTextPart(content.AsText().Text)
	case "tool_use":
		return ai.NewToolRequestPart(&ai.ToolRequest{
			Ref:   content.AsToolUse().ID,
			Name:  content.AsToolUse().Name,
			Input: content.AsToolUse().Input,
		})
	case "web_search_tool_result":
		return ai.NewCustomPart(map[string]any{
			"type": "web_search_tool_result",
			"body": anthropic.BetaWebSearchToolResultBlock{
				Content: anthropic.BetaWebSearchToolResultBlockContentUnion{
					OfBetaWebSearchResultBlockArray: content.Content.OfBetaWebSearchResultBlockArray,
					ErrorCode:                       anthropic.BetaWebSearchToolResultErrorCode(content.Content.ErrorCode),
					Type:                            constant.WebSearchToolResultError(content.Content.Type),
				},
				ToolUseID: content.ToolUseID,
			},
		})
	case "redacted_thinking":
		return ai.NewCustomPart(map[string]any{
			"type": "redacted_thinking",
			"body": content.AsRedactedThinking(),
		})
	case "thinking":
		return ai.NewReasoningPart(content.AsThinking().Thinking, []byte(content.AsThinking().Signature))
	case "server_tool_use":
		return ai.NewCustomPart(map[string]any{
			"type": "server_tool_use",
			"body": content.AsServerToolUse(),
		})
	}

	return nil
}

func translateContents(contents []anthropic.BetaContentBlockUnion) []*ai.Part {
	var parts []*ai.Part

	for _, content := range contents {
		parts = append(parts, translateContent(content))
	}

	return parts
}

func translateResponse(resp anthropic.BetaMessage, genRequest *ai.ModelRequest) (*ai.ModelResponse, error) {
	r := &ai.ModelResponse{}

	m := &ai.Message{
		Role: ai.RoleModel,
	}

	m.Content = translateContents(resp.Content)
	r.Message = m

	// Map stop reason
	switch resp.StopReason {
	case anthropic.BetaStopReasonEndTurn:
		r.FinishReason = ai.FinishReasonStop
	case anthropic.BetaStopReasonMaxTokens:
		r.FinishReason = ai.FinishReasonLength
	case anthropic.BetaStopReasonStopSequence:
		r.FinishReason = ai.FinishReasonStop
	case anthropic.BetaStopReasonToolUse:
		r.FinishReason = ai.FinishReasonStop
	default:
		if resp.StopReason != "" {
			r.FinishReason = ai.FinishReasonOther
		}
	}

	// Extract usage information
	r.Usage = &ai.GenerationUsage{
		InputTokens:  int(resp.Usage.InputTokens),
		OutputTokens: int(resp.Usage.OutputTokens),
		TotalTokens:  int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		Custom: map[string]float64{
			"cache_read_tokens":   float64(resp.Usage.CacheReadInputTokens),
			"cache_write_tokens":  float64(resp.Usage.CacheCreationInputTokens),
			"web_search_requests": float64(resp.Usage.ServerToolUse.WebSearchRequests),
		},
	}

	// Set custom data
	r.Custom = resp
	r.Request = genRequest

	return r, nil
}

func getAnthropicMediaType(mimeType string) anthropic.BetaBase64ImageSourceMediaType {
	switch strings.ToLower(mimeType) {
	case "image/jpeg", "image/jpg":
		return anthropic.BetaBase64ImageSourceMediaTypeImageJPEG
	case "image/png":
		return anthropic.BetaBase64ImageSourceMediaTypeImagePNG
	case "image/gif":
		return anthropic.BetaBase64ImageSourceMediaTypeImageGIF
	case "image/webp":
		return anthropic.BetaBase64ImageSourceMediaTypeImageWebP
	default:
		// Default to JPEG if unsupported
		return anthropic.BetaBase64ImageSourceMediaTypeImageJPEG
	}
}
