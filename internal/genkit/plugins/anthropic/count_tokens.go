package anthropic

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/pkg/errors"
)

func (a *Anthropic) CountTokens(ctx context.Context, g *genkit.Genkit, msgs []*ai.Message, docs []*ai.Document, toolDefs []ai.Tool) (int, error) {
	messages, systems, err := convertMessages(msgs, docs, true)
	if err != nil {
		return 0, err
	}

	params := anthropic.BetaMessageCountTokensParams{
		Model:    anthropic.ModelClaude4Sonnet20250514, // Default model for token counting
		Messages: messages,
		System: anthropic.BetaMessageCountTokensParamsSystemUnion{
			OfBetaTextBlockArray: systems,
		},
	}

	// Handle tools if present
	if len(toolDefs) > 0 {
		tools := make([]anthropic.BetaMessageCountTokensParamsToolUnion, len(toolDefs))
		for i, tool := range toolDefs {
			switch tool.Name() {
			case "web_search":
				tools[i] = anthropic.BetaMessageCountTokensParamsToolUnion{
					OfWebSearchTool20250305: &anthropic.BetaWebSearchTool20250305Param{
						MaxUses: anthropic.Int(99),
					},
				}
			default:
				if tool.Definition() == nil {
					return 0, errors.Errorf("tool %s has no definition", tool.Name())
				}
				tools[i] = anthropic.BetaMessageCountTokensParamsToolUnion{
					OfTool: convertTool(tool.Definition()).OfTool,
				}
			}
		}
		params.Tools = tools
	}

	count, err := a.client.Beta.Messages.CountTokens(
		ctx,
		params,
	)
	if err != nil {
		return 0, err
	}

	return int(count.InputTokens), nil
}

func CountTokens(ctx context.Context, g *genkit.Genkit, msgs []*ai.Message, docs []*ai.Document, toolDefs []ai.Tool) (int, error) {
	anthropicPlugin, ok := genkit.LookupPlugin(g, provider).(*Anthropic)
	if anthropicPlugin == nil || !ok {
		return 0, errors.Errorf("plugin %s is not a %T", provider, anthropicPlugin)
	}

	return anthropicPlugin.CountTokens(ctx, g, msgs, docs, toolDefs)
}
