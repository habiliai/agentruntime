package engine

import (
	"context"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/genkit/plugins/anthropic"
	"github.com/habiliai/agentruntime/internal/sliceutils"
	"github.com/pkg/errors"
)

func CountTokens(
	ctx context.Context,
	g *genkit.Genkit,
	promptValues *ChatPromptValues,
) (int, error) {
	msgs, err := convertToMessages(promptValues)
	if err != nil {
		return 0, errors.Wrapf(err, "failed to convert to messages")
	}

	provider := promptValues.Agent.GetModelProvider()

	switch strings.ToLower(provider) {
	case "openai", "gpt", "gpt-4", "gpt-3.5", "gpt-5":
		return 0, errors.New("openai token counter is not supported")

	case "anthropic", "claude":
		// For Anthropic, we need to know the specific model
		// Default to claude-3-5-sonnet if not specified
		return anthropic.CountTokens(ctx, g, promptValues.Agent.ModelConfig, msgs, nil, promptValues.Tools)

	default:
		return 0, fmt.Errorf("unsupported token counter provider: %s", provider)
	}
}

func (s *Engine) EstimateTokens(
	ctx context.Context,
	agent entity.Agent,
	req RunRequest,
) (int, error) {
	promptValues, err := s.BuildPromptValues(ctx, agent, req, nil)
	if err != nil {
		return 0, errors.Wrapf(err, "failed to build prompt values")
	}

	// Use conversation summarizer if available
	if s.conversationSummarizer != nil && len(req.History) > 0 {
		result, err := s.conversationSummarizer.ProcessConversationHistory(ctx, promptValues)
		if err != nil {
			return 0, err
		}

		req.History = result.RecentConversations
		promptValues, err = s.BuildPromptValues(ctx, agent, req, result.Summary)
		if err != nil {
			return 0, errors.Wrapf(err, "failed to build prompt values")
		}
	} else {
		// Fall back to simple truncation when summarizer is not available
		recentConversations := sliceutils.Cut(req.History, -200, len(req.History))
		promptValues.RecentConversations = recentConversations
	}

	return CountTokens(ctx, s.genkit, promptValues)
}
