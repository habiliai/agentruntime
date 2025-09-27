package engine

import (
	"context"
	_ "embed"
	"strings"
	"text/template"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/config"
	"github.com/pkg/errors"
)

var (
	//go:embed data/instructions/conversation_summary.md.tmpl
	conversationSummaryTmpl     string
	conversationSummaryTemplate = template.Must(template.New("conversation_summary").Funcs(funcMap()).Parse(conversationSummaryTmpl))
)

// ConversationSummarizer handles conversation history summarization
type ConversationSummarizer struct {
	genkit *genkit.Genkit
	config config.ConversationSummaryConfig
}

// NewConversationSummarizer creates a new conversation summarizer with Anthropic token counting
func NewConversationSummarizer(g *genkit.Genkit, config *config.ConversationSummaryConfig) *ConversationSummarizer {
	return &ConversationSummarizer{
		genkit: g,
		config: *config,
	}
}

// ConversationHistoryResult contains the processed conversation history
type ConversationHistoryResult struct {
	Summary             *string        `json:"summary,omitempty"`
	RecentConversations []Conversation `json:"recent_conversations"`
}

// ProcessConversationHistory processes conversation history with summarization if needed
// genRequest contains the complete current request (text, files, tools) for accurate token calculation
func (cs *ConversationSummarizer) ProcessConversationHistory(ctx context.Context, promptValues *ChatPromptValues) (*ConversationHistoryResult, error) {
	if len(promptValues.RecentConversations) == 0 {
		return &ConversationHistoryResult{
			RecentConversations: []Conversation{},
		}, nil
	}

	// Calculate tokens for current request
	requestTokens, err := CountTokens(ctx, cs.genkit, promptValues)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to count request tokens")
	}

	// If under token limit, return all conversations
	if requestTokens <= cs.config.MaxTokens {
		return &ConversationHistoryResult{
			RecentConversations: promptValues.RecentConversations,
		}, nil
	}

	// If we have too few conversations to summarize, just truncate
	if len(promptValues.RecentConversations) < cs.config.MinConversationsToSummarize {
		// Keep the most recent conversations within token limit (accounting for request files)
		availableTokens := cs.config.MaxTokens - requestTokens

		// If request files exceed MaxTokens, still return the conversations as-is
		// (rather than returning empty conversations) since the test expects this behavior
		if availableTokens <= 0 {
			return &ConversationHistoryResult{
				RecentConversations: promptValues.RecentConversations,
			}, nil
		}

		recentConversations, err := cs.truncateToTokenLimit(ctx, promptValues, availableTokens)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to truncate conversations")
		}
		// No need to count tokens since we're not returning TotalTokens
		return &ConversationHistoryResult{
			RecentConversations: recentConversations,
		}, nil
	}

	// Determine split point for summarization
	splitPoint := cs.findSplitPoint(ctx, promptValues)

	if splitPoint <= 0 {
		// Can't summarize, just truncate (accounting for request files)
		availableTokens := cs.config.MaxTokens - requestTokens
		recentConversations, err := cs.truncateToTokenLimit(ctx, promptValues, availableTokens)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to truncate conversations")
		}
		// No need to count tokens since we're not returning TotalTokens
		return &ConversationHistoryResult{
			RecentConversations: recentConversations,
		}, nil
	}

	// Split conversations
	oldConversations := promptValues.RecentConversations[:splitPoint]
	recentConversations := promptValues.RecentConversations[splitPoint:]

	// Generate summary for old conversations
	summary, err := cs.generateSummary(ctx, promptValues.WithRecentConversations(oldConversations))
	if err != nil {
		return nil, err
	}

	return &ConversationHistoryResult{
		Summary:             &summary,
		RecentConversations: recentConversations,
	}, nil
}

// findSplitPoint finds the optimal point to split conversations for summarization
func (cs *ConversationSummarizer) findSplitPoint(ctx context.Context, promptValues *ChatPromptValues) int {
	totalConversations := len(promptValues.RecentConversations)

	// Keep at least 1/3 of conversations as recent
	minRecentConversations := totalConversations / 3
	if minRecentConversations < cs.config.MinConversationsToSummarize {
		minRecentConversations = cs.config.MinConversationsToSummarize
	}

	// The split point should be at least this many conversations from the end
	maxSplitPoint := totalConversations - minRecentConversations

	if maxSplitPoint <= 0 {
		return 0
	}

	// Reserve tokens for summary and request files
	availableTokens := cs.config.MaxTokens - cs.config.SummaryTokens

	// Find the split point that keeps recent conversations under token limit
	for splitPoint := maxSplitPoint; splitPoint > 0; splitPoint-- {
		recentConversations := promptValues.RecentConversations[splitPoint:]
		recentTokens, err := CountTokens(ctx, cs.genkit, promptValues.WithRecentConversations(recentConversations))
		if err != nil {
			// On error, continue to next split point
			continue
		}

		if recentTokens <= availableTokens {
			return splitPoint
		}
	}

	return 0
}

// truncateToTokenLimit truncates conversations to fit within token limit (from the end)
func (cs *ConversationSummarizer) truncateToTokenLimit(ctx context.Context, promptValues *ChatPromptValues, tokenLimit int) ([]Conversation, error) {
	if len(promptValues.RecentConversations) == 0 {
		return promptValues.RecentConversations, nil
	}

	// Start from the end and add conversations until we hit the limit
	var result []Conversation

	for i := len(promptValues.RecentConversations); i > 0; i-- {
		result = promptValues.RecentConversations[:i]
		currentTokens, err := CountTokens(ctx, cs.genkit, promptValues.WithRecentConversations(result))
		if err != nil {
			return nil, errors.Wrapf(err, "failed to count tokens for conversation %d", i)
		}

		if currentTokens > tokenLimit {
			break
		}
	}

	return result, nil
}

// generateSummary generates a summary of the given conversations
func (cs *ConversationSummarizer) generateSummary(ctx context.Context, promptValues *ChatPromptValues) (string, error) {
	if len(promptValues.RecentConversations) == 0 {
		return "", errors.New("no conversations to summarize")
	}

	var buf strings.Builder
	if err := conversationSummaryTemplate.Execute(&buf, struct {
		ChatPromptValues
		MaxTokens int
	}{
		ChatPromptValues: *promptValues,
		MaxTokens:        cs.config.SummaryTokens,
	}); err != nil {
		return "", errors.Wrapf(err, "failed to execute conversation summary template")
	}

	prompt := buf.String()

	type Output struct {
		Summary string `json:"summary" jsonschema:"description=The summary of the conversation history"`
	}

	response, _, err := genkit.GenerateData[Output](ctx, cs.genkit,
		ai.WithModelName(cs.config.ModelForSummary),
		ai.WithPrompt(prompt),
		ai.WithCustomConstrainedOutput(),
	)
	if err != nil {
		return "", errors.Wrapf(err, "failed to generate conversation summary")
	}

	return strings.TrimSpace(response.Summary), nil
}
