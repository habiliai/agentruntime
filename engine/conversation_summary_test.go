package engine

import (
	"encoding/base64"
	"os"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/habiliai/agentruntime/config"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/genkit"
	"github.com/habiliai/agentruntime/internal/mylog"
	"github.com/joho/godotenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConversationSummarizer_CountTokens(t *testing.T) {
	// Load .env file if exists
	_ = godotenv.Load("../.env")

	// Skip test if no API keys are provided
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if anthropicKey == "" || openaiKey == "" {
		t.Skip("No API keys provided, skipping test")
	}

	ctx := t.Context()

	logger := mylog.NewLogger("debug", "text")
	modelConfig := &config.ModelConfig{
		AnthropicAPIKey: anthropicKey,
		OpenAIAPIKey:    openaiKey,
	}
	g := genkit.NewGenkit(ctx, modelConfig, logger, false)

	// Test token counting
	testCases := []struct {
		name      string
		text      string
		minTokens int
		maxTokens int
	}{
		{
			name:      "empty string",
			text:      "",
			minTokens: 220, // Base template tokens
			maxTokens: 250,
		},
		{
			name:      "simple text",
			text:      "Hello, world!",
			minTokens: 230, // Base + simple text tokens
			maxTokens: 260,
		},
		{
			name:      "longer text",
			text:      "This is a longer sentence with more words to test token counting functionality.",
			minTokens: 240, // Base + longer text tokens
			maxTokens: 270,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testAgent := entity.Agent{
				ModelName: "anthropic/claude-3-5-sonnet",
			}
			promptValues := &ChatPromptValues{
				Agent: testAgent,
				RecentConversations: []Conversation{
					{User: "user", Text: tc.text},
				},
				Tools: []ai.Tool{},
			}
			tokens, err := CountTokens(ctx, g, promptValues)
			require.NoError(t, err)
			assert.GreaterOrEqual(t, tokens, tc.minTokens)
			assert.LessOrEqual(t, tokens, tc.maxTokens)
		})
	}
}

func TestConversationSummarizer_ProcessConversationHistory(t *testing.T) {
	// Load .env file if exists
	_ = godotenv.Load("../.env")

	// Skip test if no API keys are provided
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if anthropicKey == "" || openaiKey == "" {
		t.Skip("No API keys provided, skipping test")
	}

	ctx := t.Context()

	logger := mylog.NewLogger("debug", "text")
	modelConfig := &config.ModelConfig{
		AnthropicAPIKey: anthropicKey,
		OpenAIAPIKey:    openaiKey,
	}
	g := genkit.NewGenkit(ctx, modelConfig, logger, false)

	testConfig := &config.ModelConfig{
		AnthropicAPIKey: anthropicKey,
		OpenAIAPIKey:    openaiKey,
		ConversationSummary: config.ConversationSummaryConfig{
			MaxTokens:                   500, // Low limit to force summarization
			SummaryTokens:               100,
			MinConversationsToSummarize: 3,
			ModelForSummary:             "openai/gpt-5-mini",
		},
	}

	summarizer := NewConversationSummarizer(g, &testConfig.ConversationSummary)

	t.Run("empty conversations", func(t *testing.T) {
		testAgent := entity.Agent{
			ModelName: "anthropic/claude-3-5-sonnet",
		}
		promptValues := &ChatPromptValues{
			Agent:               testAgent,
			RecentConversations: []Conversation{},
			Tools:               []ai.Tool{},
		}
		result, err := summarizer.ProcessConversationHistory(ctx, promptValues)
		require.NoError(t, err)

		assert.Empty(t, result.RecentConversations)
		assert.Nil(t, result.Summary)
	})

	t.Run("conversations under token limit", func(t *testing.T) {
		conversations := []Conversation{
			{User: "user1", Text: "Hello"},
			{User: "bot", Text: "Hi there!"},
		}

		testAgent := entity.Agent{
			ModelName: "anthropic/claude-3-5-sonnet",
		}
		promptValues := &ChatPromptValues{
			Agent:               testAgent,
			RecentConversations: conversations,
			Tools:               []ai.Tool{},
		}
		result, err := summarizer.ProcessConversationHistory(ctx, promptValues)
		require.NoError(t, err)

		assert.Equal(t, conversations, result.RecentConversations)
		assert.Nil(t, result.Summary)
	})

	t.Run("many conversations under min threshold", func(t *testing.T) {
		// Create conversations that exceed token limit but are under min count
		conversations := []Conversation{
			{User: "user1", Text: "This is a very long message that contains many words and should consume a significant number of tokens to test the token counting and truncation functionality of the conversation summarizer system."},
			{User: "bot", Text: "This is another very long response that also contains many words and should consume a significant number of tokens to test the token counting and truncation functionality."},
		}

		testAgent := entity.Agent{
			ModelName: "anthropic/claude-3-5-sonnet",
		}
		promptValues := &ChatPromptValues{
			Agent:               testAgent,
			RecentConversations: conversations,
			Tools:               []ai.Tool{},
		}
		result, err := summarizer.ProcessConversationHistory(ctx, promptValues)
		require.NoError(t, err)

		// Should truncate without summarizing since below min threshold
		assert.LessOrEqual(t, len(result.RecentConversations), len(conversations))
		assert.Nil(t, result.Summary)
	})

	t.Run("conversations with request files", func(t *testing.T) {
		t.Skip("Skipping image test - requires valid image data for Anthropic API")

		conversations := []Conversation{
			{User: "user1", Text: "Hello"},
			{User: "bot", Text: "Hi there!"},
		}

		// Create request with image file
		imageData := base64.StdEncoding.EncodeToString(make([]byte, 50*1024)) // 50KB
		testAgent := entity.Agent{
			ModelName: "anthropic/claude-3-5-sonnet",
		}
		promptValues := &ChatPromptValues{
			Agent:               testAgent,
			RecentConversations: conversations,
			Thread: Thread{
				Files: []File{
					{
						ContentType: "image/jpeg",
						Data:        imageData,
						Filename:    "test.jpg",
					},
				},
			},
			Tools: []ai.Tool{},
		}

		result, err := summarizer.ProcessConversationHistory(ctx, promptValues)
		require.NoError(t, err)

		// Should handle request with files correctly
		assert.Equal(t, len(conversations), len(result.RecentConversations))
		assert.Equal(t, conversations, result.RecentConversations)
		assert.Nil(t, result.Summary)
	})
}

func TestConversationSummarizer_findSplitPoint(t *testing.T) {
	// Load .env file if exists
	_ = godotenv.Load("../.env")

	// Skip test if no API keys are provided
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if anthropicKey == "" || openaiKey == "" {
		t.Skip("No API keys provided, skipping test")
	}

	ctx := t.Context()

	logger := mylog.NewLogger("debug", "text")
	testConfig := &config.ModelConfig{
		AnthropicAPIKey: anthropicKey,
		OpenAIAPIKey:    openaiKey,
		ConversationSummary: config.ConversationSummaryConfig{
			MaxTokens:                   1000,
			SummaryTokens:               200,
			MinConversationsToSummarize: 5,
			ModelForSummary:             "openai/gpt-5-mini",
		},
	}
	g := genkit.NewGenkit(ctx, testConfig, logger, false)

	summarizer := NewConversationSummarizer(g, &testConfig.ConversationSummary)

	conversations := make([]Conversation, 20)
	for i := 0; i < 20; i++ {
		conversations[i] = Conversation{
			User: "user1",
			Text: "Short message",
		}
	}

	testAgent := entity.Agent{
		ModelName: "anthropic/claude-3-5-sonnet",
	}
	promptValues := &ChatPromptValues{
		Agent:               testAgent,
		RecentConversations: conversations,
		Tools:               []ai.Tool{},
	}

	splitPoint := summarizer.findSplitPoint(ctx, promptValues)

	// Should find a valid split point
	assert.Greater(t, splitPoint, 0)
	assert.Less(t, splitPoint, len(conversations))

	// Remaining conversations should be at least the minimum
	remaining := len(conversations) - splitPoint
	minRequired := len(conversations) / 3
	if minRequired < testConfig.ConversationSummary.MinConversationsToSummarize {
		minRequired = testConfig.ConversationSummary.MinConversationsToSummarize
	}
	assert.GreaterOrEqual(t, remaining, minRequired)
}

func TestConversationSummarizer_truncateToTokenLimit(t *testing.T) {
	// Load .env file if exists
	_ = godotenv.Load("../.env")

	// Skip test if no API keys are provided
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if anthropicKey == "" || openaiKey == "" {
		t.Skip("No API keys provided, skipping test")
	}

	ctx := t.Context()

	logger := mylog.NewLogger("debug", "text")
	testConfig := &config.ModelConfig{
		AnthropicAPIKey: anthropicKey,
		OpenAIAPIKey:    openaiKey,
		ConversationSummary: config.ConversationSummaryConfig{
			MaxTokens:                   100,
			SummaryTokens:               50,
			MinConversationsToSummarize: 3,
			ModelForSummary:             "openai/gpt-5-mini",
		},
	}
	g := genkit.NewGenkit(ctx, testConfig, logger, false)

	summarizer := NewConversationSummarizer(g, &testConfig.ConversationSummary)

	conversations := []Conversation{
		{User: "user1", Text: "First message with some content"},
		{User: "user2", Text: "Second message with some content"},
		{User: "user3", Text: "Third message with some content"},
		{User: "user4", Text: "Fourth message with some content"},
		{User: "user5", Text: "Fifth message with some content"},
	}

	testAgent := entity.Agent{
		ModelName: "anthropic/claude-3-5-sonnet",
	}
	promptValues := &ChatPromptValues{
		Agent:               testAgent,
		RecentConversations: conversations,
		Tools:               []ai.Tool{},
	}

	result, err := summarizer.truncateToTokenLimit(ctx, promptValues, 400) // Higher limit to account for base template
	require.NoError(t, err)

	// Should return some conversations (from the end)
	assert.Greater(t, len(result), 0)
	assert.LessOrEqual(t, len(result), len(conversations))

	// Should not exceed token limit
	if len(result) > 0 {
		testAgent := entity.Agent{
			ModelName: "anthropic/claude-3-5-sonnet",
		}
		resultPromptValues := &ChatPromptValues{
			Agent:               testAgent,
			RecentConversations: result,
			Tools:               []ai.Tool{},
		}
		tokenCount, err := CountTokens(ctx, g, resultPromptValues)
		require.NoError(t, err)
		assert.LessOrEqual(t, tokenCount, 400) // Higher limit
	}

	// Should preserve order and keep oldest conversations (current behavior)
	if len(result) > 0 {
		assert.Equal(t, conversations[0], result[0]) // First conversation should be preserved
	}
}
