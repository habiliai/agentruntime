package openaiapi

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
)

func TestConvertPart(t *testing.T) {
	tests := []struct {
		name  string
		input *ai.Part
		want  goopenai.ChatCompletionContentPartUnionParam
	}{
		{
			name:  "text part",
			input: ai.NewTextPart("hi"),
			want: goopenai.ChatCompletionContentPartTextParam{
				Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
				Text: goopenai.F("hi"),
			},
		},
		{
			name:  "media part",
			input: ai.NewMediaPart("image/jpeg", "https://example.com/image.jpg"),
			want: goopenai.ChatCompletionContentPartImageParam{
				Type: goopenai.F(goopenai.ChatCompletionContentPartImageTypeImageURL),
				ImageURL: goopenai.F(goopenai.ChatCompletionContentPartImageImageURLParam{
					URL:    goopenai.F("https://example.com/image.jpg"),
					Detail: goopenai.F(goopenai.ChatCompletionContentPartImageImageURLDetailAuto),
				}),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertPart(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertPart() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestConvertMessages(t *testing.T) {
	tests := []struct {
		name  string
		input []*ai.Message
		want  []goopenai.ChatCompletionMessageParamUnion
	}{
		{
			name: "tool request",
			input: []*ai.Message{
				{
					Role: ai.RoleModel,
					Content: []*ai.Part{ai.NewToolRequestPart(
						&ai.ToolRequest{
							Name:  "tellAFunnyJoke",
							Input: json.RawMessage(`{"topic":"bob"}`),
							Ref:   "call_1234",
						},
					)},
				},
			},
			want: []goopenai.ChatCompletionMessageParamUnion{
				goopenai.ChatCompletionAssistantMessageParam{
					Role: goopenai.F(goopenai.ChatCompletionAssistantMessageParamRoleAssistant),
					ToolCalls: goopenai.F([]goopenai.ChatCompletionMessageToolCallParam{
						{
							ID:   goopenai.String("call_1234"),
							Type: goopenai.F(goopenai.ChatCompletionMessageToolCallTypeFunction),
							Function: goopenai.F(goopenai.ChatCompletionMessageToolCallFunctionParam{
								Name:      goopenai.F("tellAFunnyJoke"),
								Arguments: goopenai.F("{\"topic\":\"bob\"}"),
							}),
						},
					}),
				},
			},
		},
		{
			name: "tool response",
			input: []*ai.Message{
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{ai.NewToolResponsePart(
						&ai.ToolResponse{
							Ref:  "call_1234",
							Name: "tellAFunnyJoke",
							Output: map[string]any{
								"joke": "Why did the bob cross the road?",
							},
						},
					)},
				},
			},
			want: []goopenai.ChatCompletionMessageParamUnion{
				goopenai.ChatCompletionToolMessageParam{
					Role: goopenai.F(goopenai.ChatCompletionToolMessageParamRoleTool),
					Content: goopenai.F([]goopenai.ChatCompletionContentPartTextParam{
						{
							Text: goopenai.F("{\"joke\":\"Why did the bob cross the road?\"}"),
							Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
						},
					}),
					ToolCallID: goopenai.String("call_1234"),
				},
			},
		},
		{
			name: "text",
			input: []*ai.Message{
				{
					Role:    ai.RoleUser,
					Content: []*ai.Part{ai.NewTextPart("hi")},
				},
				{
					Role:    ai.RoleModel,
					Content: []*ai.Part{ai.NewTextPart("how can I help you?")},
				},
				{
					Role:    ai.RoleUser,
					Content: []*ai.Part{ai.NewTextPart("I am testing")},
				},
			},
			want: []goopenai.ChatCompletionMessageParamUnion{
				goopenai.ChatCompletionUserMessageParam{
					Role: goopenai.F(goopenai.ChatCompletionUserMessageParamRoleUser),
					Content: goopenai.F([]goopenai.ChatCompletionContentPartUnionParam{
						goopenai.ChatCompletionContentPartTextParam{
							Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
							Text: goopenai.F("hi"),
						},
					}),
				},
				goopenai.ChatCompletionAssistantMessageParam{
					Role: goopenai.F(goopenai.ChatCompletionAssistantMessageParamRoleAssistant),
					Content: goopenai.F([]goopenai.ChatCompletionAssistantMessageParamContentUnion{
						goopenai.ChatCompletionContentPartTextParam{
							Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
							Text: goopenai.F("how can I help you?"),
						},
					}),
				},
				goopenai.ChatCompletionUserMessageParam{
					Role: goopenai.F(goopenai.ChatCompletionUserMessageParamRoleUser),
					Content: goopenai.F([]goopenai.ChatCompletionContentPartUnionParam{
						goopenai.ChatCompletionContentPartTextParam{
							Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
							Text: goopenai.F("I am testing"),
						},
					}),
				},
			},
		},
		{
			name: "multi-modal (text + media)",
			input: []*ai.Message{
				{
					Role: ai.RoleUser,
					Content: []*ai.Part{
						ai.NewTextPart("describe the following image:"),
						ai.NewMediaPart("image/jpeg", "https://example.com/image.jpg"),
					},
				},
			},
			want: []goopenai.ChatCompletionMessageParamUnion{
				goopenai.ChatCompletionUserMessageParam{
					Role: goopenai.F(goopenai.ChatCompletionUserMessageParamRoleUser),
					Content: goopenai.F([]goopenai.ChatCompletionContentPartUnionParam{
						goopenai.ChatCompletionContentPartTextParam{
							Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
							Text: goopenai.F("describe the following image:"),
						},
						goopenai.ChatCompletionContentPartImageParam{
							Type: goopenai.F(goopenai.ChatCompletionContentPartImageTypeImageURL),
							ImageURL: goopenai.F(goopenai.ChatCompletionContentPartImageImageURLParam{
								URL:    goopenai.F("https://example.com/image.jpg"),
								Detail: goopenai.F(goopenai.ChatCompletionContentPartImageImageURLDetailAuto),
							}),
							//goopenai.F("describe the following image:"),
						},
					}),
				},
			},
		},
		{
			name: "system message",
			input: []*ai.Message{
				{
					Role:    ai.RoleSystem,
					Content: []*ai.Part{ai.NewTextPart("system message")},
				},
			},
			want: []goopenai.ChatCompletionMessageParamUnion{
				goopenai.ChatCompletionSystemMessageParam{
					Role: goopenai.F(goopenai.ChatCompletionSystemMessageParamRoleSystem),
					Content: goopenai.F([]goopenai.ChatCompletionContentPartTextParam{
						{
							Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
							Text: goopenai.F("system message"),
						},
					}),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertMessages(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestConvertToolCall(t *testing.T) {
	tests := []struct {
		name  string
		input *ai.Part
		want  goopenai.ChatCompletionMessageToolCallParam
	}{
		{
			name: "tool call",
			input: ai.NewToolRequestPart(
				&ai.ToolRequest{
					Name: "tellAFunnyJoke",
					Input: map[string]any{
						"topic": "bob",
					},
					Ref: "call_1234",
				},
			),
			want: goopenai.ChatCompletionMessageToolCallParam{
				ID:   goopenai.String("call_1234"),
				Type: goopenai.F(goopenai.ChatCompletionMessageToolCallTypeFunction),
				Function: goopenai.F(goopenai.ChatCompletionMessageToolCallFunctionParam{
					Name:      goopenai.F("tellAFunnyJoke"),
					Arguments: goopenai.F("{\"topic\":\"bob\"}"),
				}),
			},
		},
		{
			name: "tool call with empty input",
			input: ai.NewToolRequestPart(
				&ai.ToolRequest{
					Name:  "tellAFunnyJoke",
					Input: map[string]any{},
					Ref:   "call_1234",
				},
			),
			want: goopenai.ChatCompletionMessageToolCallParam{
				ID:   goopenai.String("call_1234"),
				Type: goopenai.F(goopenai.ChatCompletionMessageToolCallTypeFunction),
				Function: goopenai.F(goopenai.ChatCompletionMessageToolCallFunctionParam{
					Name:      goopenai.F("tellAFunnyJoke"),
					Arguments: goopenai.F("{}"),
				}),
			},
		},
		{
			name: "tool call with nil input",
			input: ai.NewToolRequestPart(
				&ai.ToolRequest{
					Name:  "tellAFunnyJoke",
					Input: nil,
					Ref:   "call_1234",
				},
			),
			want: goopenai.ChatCompletionMessageToolCallParam{
				ID:   goopenai.String("call_1234"),
				Type: goopenai.F(goopenai.ChatCompletionMessageToolCallTypeFunction),
				Function: goopenai.F(goopenai.ChatCompletionMessageToolCallFunctionParam{
					Name:      goopenai.F("tellAFunnyJoke"),
					Arguments: goopenai.F("{}"),
				}),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertToolCall(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertToolCall() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestConvertTool(t *testing.T) {
	tests := []struct {
		name  string
		input *ai.ToolDefinition
		want  goopenai.ChatCompletionToolParam
	}{
		{
			name: "text part",
			input: &ai.ToolDefinition{
				Name:        "tellAFunnyJoke",
				Description: "use when want to tell a funny joke",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"topic": map[string]any{
							"type": "string",
						},
					},
					"required":             []string{"topic"},
					"additionalProperties": false,
					"$schema":              "http://json-schema.org/draft-07/schema#",
				},
				OutputSchema: map[string]any{
					"type":    "string",
					"$schema": "http://json-schema.org/draft-07/schema#",
				},
			},
			want: goopenai.ChatCompletionToolParam{
				Type: goopenai.F(goopenai.ChatCompletionToolTypeFunction),
				Function: goopenai.F(shared.FunctionDefinitionParam{
					Name:        goopenai.F("tellAFunnyJoke"),
					Description: goopenai.F("use when want to tell a funny joke"),
					Strict:      goopenai.F(false),
					Parameters: goopenai.F(shared.FunctionParameters{
						"type": "object",
						"properties": map[string]any{
							"topic": map[string]any{
								"type": "string",
							},
						},
						"required":             []string{"topic"},
						"additionalProperties": false,
						"$schema":              "http://json-schema.org/draft-07/schema#",
					}),
				}),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertTool(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertTool() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestConvertRequest(t *testing.T) {
	tests := []struct {
		name  string
		input struct {
			model string
			req   *ai.ModelRequest
		}
		want goopenai.ChatCompletionNewParams
	}{
		{
			name: "request with text messages",
			input: struct {
				model string
				req   *ai.ModelRequest
			}{
				model: goopenai.ChatModelGPT4o,
				req: &ai.ModelRequest{
					Messages: []*ai.Message{
						{
							Role:    ai.RoleUser,
							Content: []*ai.Part{ai.NewTextPart("Tell a joke about dogs.")},
						},
					},
					Config: &ai.GenerationCommonConfig{
						MaxOutputTokens: 10,
						StopSequences:   []string{"\n"},
						Temperature:     0.7,
						TopP:            0.9,
					},
					Output: &ai.ModelOutputConfig{
						Format: ai.OutputFormatText,
					},
				},
			},
			want: goopenai.ChatCompletionNewParams{
				Model: goopenai.F(goopenai.ChatModelGPT4o),
				Messages: goopenai.F([]goopenai.ChatCompletionMessageParamUnion{
					goopenai.ChatCompletionUserMessageParam{
						Role: goopenai.F(goopenai.ChatCompletionUserMessageParamRoleUser),
						Content: goopenai.F([]goopenai.ChatCompletionContentPartUnionParam{
							goopenai.ChatCompletionContentPartTextParam{
								Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
								Text: goopenai.F("Tell a joke about dogs."),
							},
						}),
					},
				}),
				ResponseFormat: goopenai.F[goopenai.ChatCompletionNewParamsResponseFormatUnion](
					goopenai.ResponseFormatTextParam{
						Type: goopenai.F(goopenai.ResponseFormatTextTypeText),
					},
				),
				MaxTokens:   goopenai.F[int64](10),
				Stop:        goopenai.F[goopenai.ChatCompletionNewParamsStopUnion](goopenai.ChatCompletionNewParamsStopArray([]string{"\n"})),
				Temperature: goopenai.F(0.7),
				TopP:        goopenai.F(0.9),
			},
		},
		{
			name: "request with text messages and tools",
			input: struct {
				model string
				req   *ai.ModelRequest
			}{
				model: goopenai.ChatModelGPT4o,
				req: &ai.ModelRequest{
					Messages: []*ai.Message{
						{
							Role:    ai.RoleUser,
							Content: []*ai.Part{ai.NewTextPart("Tell a joke about dogs.")},
						},
						{
							Role: ai.RoleModel,
							Content: []*ai.Part{ai.NewToolRequestPart(
								&ai.ToolRequest{
									Name: "tellAFunnyJoke",
									Input: map[string]any{
										"topic": "dogs",
									},
									Ref: "call_1234",
								},
							)},
						},
						{
							Role: ai.RoleTool,
							Content: []*ai.Part{ai.NewToolResponsePart(
								&ai.ToolResponse{
									Name: "tellAFunnyJoke",
									Output: map[string]any{
										"joke": "Why did the dogs cross the road?",
									},
									Ref: "call_1234",
								},
							)},
						},
					},
					Tools: []*ai.ToolDefinition{
						{
							Name:        "tellAFunnyJoke",
							Description: "Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.",
							InputSchema: map[string]any{
								"type": "object",
								"properties": map[string]any{
									"topic": map[string]any{
										"type": "string",
									},
								},
								"required":             []string{"topic"},
								"additionalProperties": false,
								"$schema":              "http://json-schema.org/draft-07/schema#",
							},
							OutputSchema: map[string]any{
								"type":    "string",
								"$schema": "http://json-schema.org/draft-07/schema#",
							},
						},
					},
					Output: &ai.ModelOutputConfig{
						Format: ai.OutputFormatText,
					},
				},
			},
			want: goopenai.ChatCompletionNewParams{
				Model: goopenai.F(goopenai.ChatModelGPT4o),
				Messages: goopenai.F([]goopenai.ChatCompletionMessageParamUnion{
					goopenai.ChatCompletionUserMessageParam{
						Role: goopenai.F(goopenai.ChatCompletionUserMessageParamRoleUser),
						Content: goopenai.F([]goopenai.ChatCompletionContentPartUnionParam{
							goopenai.ChatCompletionContentPartTextParam{
								Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
								Text: goopenai.F("Tell a joke about dogs."),
							},
						}),
					},
					goopenai.ChatCompletionAssistantMessageParam{
						Role: goopenai.F(goopenai.ChatCompletionAssistantMessageParamRoleAssistant),
						ToolCalls: goopenai.F([]goopenai.ChatCompletionMessageToolCallParam{
							{
								ID:   goopenai.String("call_1234"),
								Type: goopenai.F(goopenai.ChatCompletionMessageToolCallTypeFunction),
								Function: goopenai.F(goopenai.ChatCompletionMessageToolCallFunctionParam{
									Name:      goopenai.F("tellAFunnyJoke"),
									Arguments: goopenai.F("{\"topic\":\"dogs\"}"),
								}),
							},
						}),
					},
					goopenai.ChatCompletionToolMessageParam{
						Role: goopenai.F(goopenai.ChatCompletionToolMessageParamRoleTool),
						Content: goopenai.F([]goopenai.ChatCompletionContentPartTextParam{
							{
								Text: goopenai.F("{\"joke\":\"Why did the dogs cross the road?\"}"),
								Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
							},
						}),
						ToolCallID: goopenai.String("call_1234"),
					},
				}),
				Tools: goopenai.F([]goopenai.ChatCompletionToolParam{
					{
						Type: goopenai.F(goopenai.ChatCompletionToolTypeFunction),
						Function: goopenai.F(shared.FunctionDefinitionParam{
							Name:        goopenai.F("tellAFunnyJoke"),
							Description: goopenai.F("Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke."),
							Parameters: goopenai.F(shared.FunctionParameters{
								"type": "object",
								"properties": map[string]any{
									"topic": map[string]any{
										"type": "string",
									},
								},
								"required":             []string{"topic"},
								"additionalProperties": false,
								"$schema":              "http://json-schema.org/draft-07/schema#",
							}),
							Strict: goopenai.F(false),
						}),
					},
				}),
				ResponseFormat: goopenai.F[goopenai.ChatCompletionNewParamsResponseFormatUnion](
					goopenai.ResponseFormatTextParam{
						Type: goopenai.F(goopenai.ResponseFormatTextTypeText),
					},
				),
			},
		},
		{
			name: "request with structured output: json",
			input: struct {
				model string
				req   *ai.ModelRequest
			}{
				model: goopenai.ChatModelGPT4o,
				req: &ai.ModelRequest{
					Messages: []*ai.Message{
						{
							Role:    ai.RoleUser,
							Content: []*ai.Part{ai.NewTextPart("Tell a joke about dogs.")},
						},
						{
							Role: ai.RoleModel,
							Content: []*ai.Part{ai.NewToolRequestPart(
								&ai.ToolRequest{
									Name: "tellAFunnyJoke",
									Input: map[string]any{
										"topic": "dogs",
									},
									Ref: "call_1234",
								},
							)},
						},
						{
							Role: ai.RoleTool,
							Content: []*ai.Part{ai.NewToolResponsePart(
								&ai.ToolResponse{
									Name: "tellAFunnyJoke",
									Output: map[string]any{
										"joke": "Why did the dogs cross the road?",
									},
									Ref: "call_1234",
								},
							)},
						},
					},
					Tools: []*ai.ToolDefinition{
						{
							Name:        "tellAFunnyJoke",
							Description: "Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.",
							InputSchema: map[string]any{
								"type": "object",
								"properties": map[string]any{
									"topic": map[string]any{
										"type": "string",
									},
								},
								"required":             []string{"topic"},
								"additionalProperties": false,
								"$schema":              "http://json-schema.org/draft-07/schema#",
							},
							OutputSchema: map[string]any{
								"type":    "string",
								"$schema": "http://json-schema.org/draft-07/schema#",
							},
						},
					},
					Output: &ai.ModelOutputConfig{
						Format: ai.OutputFormatJSON,
					},
				},
			},
			want: goopenai.ChatCompletionNewParams{
				Model: goopenai.F(goopenai.ChatModelGPT4o),
				Messages: goopenai.F([]goopenai.ChatCompletionMessageParamUnion{
					goopenai.ChatCompletionUserMessageParam{
						Role: goopenai.F(goopenai.ChatCompletionUserMessageParamRoleUser),
						Content: goopenai.F([]goopenai.ChatCompletionContentPartUnionParam{
							goopenai.ChatCompletionContentPartTextParam{
								Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
								Text: goopenai.F("Tell a joke about dogs."),
							},
						}),
					},
					goopenai.ChatCompletionAssistantMessageParam{
						Role: goopenai.F(goopenai.ChatCompletionAssistantMessageParamRoleAssistant),
						ToolCalls: goopenai.F([]goopenai.ChatCompletionMessageToolCallParam{
							{
								ID:   goopenai.String("call_1234"),
								Type: goopenai.F(goopenai.ChatCompletionMessageToolCallTypeFunction),
								Function: goopenai.F(goopenai.ChatCompletionMessageToolCallFunctionParam{
									Name:      goopenai.F("tellAFunnyJoke"),
									Arguments: goopenai.F("{\"topic\":\"dogs\"}"),
								}),
							},
						}),
					},
					goopenai.ChatCompletionToolMessageParam{
						Role: goopenai.F(goopenai.ChatCompletionToolMessageParamRoleTool),
						Content: goopenai.F([]goopenai.ChatCompletionContentPartTextParam{
							{
								Text: goopenai.F("{\"joke\":\"Why did the dogs cross the road?\"}"),
								Type: goopenai.F(goopenai.ChatCompletionContentPartTextTypeText),
							},
						}),
						ToolCallID: goopenai.String("call_1234"),
					},
				}),
				Tools: goopenai.F([]goopenai.ChatCompletionToolParam{
					{
						Type: goopenai.F(goopenai.ChatCompletionToolTypeFunction),
						Function: goopenai.F(shared.FunctionDefinitionParam{
							Name:        goopenai.F("tellAFunnyJoke"),
							Description: goopenai.F("Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke."),
							Parameters: goopenai.F(shared.FunctionParameters{
								"type": "object",
								"properties": map[string]any{
									"topic": map[string]any{
										"type": "string",
									},
								},
								"required":             []string{"topic"},
								"additionalProperties": false,
								"$schema":              "http://json-schema.org/draft-07/schema#",
							}),
							Strict: goopenai.F(false),
						}),
					},
				}),
				ResponseFormat: goopenai.F[goopenai.ChatCompletionNewParamsResponseFormatUnion](
					goopenai.ResponseFormatJSONObjectParam{
						Type: goopenai.F(goopenai.ResponseFormatJSONObjectTypeJSONObject),
					},
				),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertRequest(tt.input.model, tt.input.req)
			if err != nil {
				t.Fatal(err)
			}
			assert.Equal(t, tt.want, got)
		})
	}
}
