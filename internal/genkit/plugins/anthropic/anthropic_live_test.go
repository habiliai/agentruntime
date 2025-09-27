package anthropic_test

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/internal/genkit/plugins/anthropic"
	"github.com/mokiat/gog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

//go:embed testdata/red_image.png
var redImageFile []byte

func TestLive_GenerateText(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	tests := []struct {
		name      string
		modelName string
		prompt    string
		timeout   time.Duration
	}{
		{
			name:      "claude-3.5-haiku simple",
			modelName: "claude-3.5-haiku",
			prompt:    "What is 2+2? Answer with just the number.",
			timeout:   30 * time.Minute,
		},
		{
			name:      "claude-3.7-sonnet simple",
			modelName: "claude-3.7-sonnet",
			prompt:    "What is the capital of Japan? Answer with just the city name.",
			timeout:   30 * time.Minute,
		},
		{
			name:      "claude-4-sonnet simple",
			modelName: "claude-4-sonnet",
			prompt:    "What is the capital of France? Answer with just the city name.",
			timeout:   30 * time.Minute,
		},
		{
			name:      "claude-3.5-haiku with math",
			modelName: "claude-3.5-haiku",
			prompt:    "What is 10 divided by 2? Answer with just the number.",
			timeout:   30 * time.Minute,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := anthropic.Model(g, tt.modelName)
			if model == nil {
				t.Skipf("Model %s not available", tt.modelName)
			}

			// Create context with timeout
			ctx, cancel := context.WithTimeoutCause(
				context.Background(),
				tt.timeout,
				fmt.Errorf("timeout %s", tt.modelName),
			)
			defer cancel()

			req := &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							ai.NewTextPart(tt.prompt),
						},
					},
				},
				Config: &ai.GenerationCommonConfig{
					MaxOutputTokens: 100,
					Temperature:     0.0,
				},
			}

			resp, err := model.Generate(ctx, req, nil)
			require.NoError(t, err)
			require.NotNil(t, resp)
			require.NotNil(t, resp.Message)
			require.NotEmpty(t, resp.Message.Content)

			content := resp.Message.Content[0].Text
			assert.NotEmpty(t, content)
			fmt.Printf("%s response: %s\n", tt.modelName, content)
		})
	}
}

func TestLive_GenerateWithStreaming(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.TODO()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	model := anthropic.Model(g, "claude-3.5-haiku")
	require.NotNil(t, model)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("Count from 1 to 5. Just the numbers, one per line."),
				},
			},
		},
		Config: &ai.GenerationCommonConfig{
			MaxOutputTokens: 50,
		},
	}

	var chunks []string
	resp, err := model.Generate(ctx, req, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		if len(chunk.Content) > 0 && chunk.Content[0].Text != "" {
			chunks = append(chunks, chunk.Content[0].Text)
			fmt.Print(chunk.Content[0].Text)
		}
		return nil
	})
	fmt.Println()

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.NotEmpty(t, chunks)
	assert.NotEmpty(t, resp.Message.Content[0].Text)
}

func TestLive_GenerateWithImage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	model := anthropic.Model(g, "claude-3.5-haiku")
	require.NotNil(t, model)

	// Create a simple base64 encoded image (200x200 red pixel PNG)
	base64Image := base64.StdEncoding.EncodeToString(redImageFile)

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("What color is this image? Answer with just the color name."),
					ai.NewMediaPart("image/png", base64Image),
				},
			},
		},
		Config: &ai.GenerationCommonConfig{
			MaxOutputTokens: 50,
			Temperature:     0.0,
		},
	}

	resp, err := model.Generate(ctx, req, nil)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Message)

	content := resp.Message.Content[0].Text
	assert.Contains(t, strings.ToLower(content), "red", "Expected the model to identify the red pixel")
	fmt.Printf("Image analysis response: %s\n", content)
}

func TestLive_GenerateWithSystemMessage(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.TODO()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	model := anthropic.Model(g, "claude-3.5-haiku")
	require.NotNil(t, model)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleSystem,
				Content: []*ai.Part{
					ai.NewTextPart("You are a helpful assistant that always responds in haiku format."),
				},
			},
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("Tell me about the ocean."),
				},
			},
		},
		Config: &ai.GenerationCommonConfig{
			MaxOutputTokens: 100,
		},
	}

	resp, err := model.Generate(ctx, req, nil)
	require.NoError(t, err)
	require.NotNil(t, resp)

	content := resp.Message.Content[0].Text
	assert.NotEmpty(t, content)
	fmt.Printf("Haiku response: %s\n", content)
}

// TestLive_GenerateWithReasoning tests the extended thinking (reasoning) feature
// which allows models to show their step-by-step thought process.
// This feature is especially useful for complex problems that require logical reasoning.
func TestLive_GenerateWithReasoning(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	tests := []struct {
		name        string
		modelName   string
		prompt      string
		enabled     bool
		budgetRatio float64
		timeout     time.Duration
	}{
		{
			name:        "claude-3.5-haiku without reasoning",
			modelName:   "claude-3.5-haiku",
			prompt:      "If a train travels 120 miles in 2 hours, what is its speed in mph? Think step by step.",
			enabled:     false,
			budgetRatio: 0,
			timeout:     30 * time.Minute,
		},
		{
			name:        "claude-3.7-sonnet with reasoning enabled",
			modelName:   "claude-3.7-sonnet",
			prompt:      "A farmer has 17 sheep. All but 9 die. How many are left? Think step by step.",
			enabled:     true,
			budgetRatio: 0.25, // 25% of maxOutputTokens
			timeout:     30 * time.Minute,
		},
		{
			name:        "claude-4-sonnet with reasoning enabled",
			modelName:   "claude-4-sonnet",
			prompt:      "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think carefully.",
			enabled:     true,
			budgetRatio: 0.25, // 25% of maxOutputTokens
			timeout:     30 * time.Minute,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := anthropic.Model(g, tt.modelName)
			if model == nil {
				t.Skipf("Model %s not available", tt.modelName)
			}

			// Create context with timeout
			ctx, cancel := context.WithTimeoutCause(
				context.Background(),
				tt.timeout,
				fmt.Errorf("timeout %s", tt.modelName),
			)
			defer cancel()

			config := struct {
				ai.GenerationCommonConfig
				anthropic.ExtendedThinkingConfig
			}{
				GenerationCommonConfig: ai.GenerationCommonConfig{
					MaxOutputTokens: 8192, // Large enough to support reasoning budget (8192 * 0.15 = 1228 > 1024, leaving 6963 for actual output)
					Temperature:     0.0,
				},
				ExtendedThinkingConfig: anthropic.ExtendedThinkingConfig{
					ExtendedThinkingEnabled:     tt.enabled,
					ExtendedThinkingBudgetRatio: tt.budgetRatio,
				},
			}

			req := &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							ai.NewTextPart(tt.prompt),
						},
					},
				},
				Config: config,
			}

			resp, err := model.Generate(ctx, req, nil)
			require.NoError(t, err)
			require.NotNil(t, resp)
			require.NotNil(t, resp.Message)
			require.NotEmpty(t, resp.Message.Content)

			// Check for reasoning part
			hasReasoning := false
			var reasoningText string
			var finalAnswer string

			for _, part := range resp.Message.Content {
				if part.IsReasoning() {
					hasReasoning = true
					reasoningText = part.Text
					t.Logf("Found reasoning part with %d characters", len(reasoningText))
				} else if part.IsText() {
					finalAnswer = part.Text
				}
			}

			// claude-3.5-haiku doesn't support reasoning, so skip assertions for it
			if tt.modelName != "claude-3.5-haiku" {
				if tt.enabled && tt.budgetRatio > 0 {
					assert.True(t, hasReasoning, "Expected reasoning part to be present when reasoning is enabled")
					assert.NotEmpty(t, reasoningText, "Reasoning text should not be empty")
					t.Logf("Reasoning: %s", reasoningText)
				} else if !tt.enabled {
					assert.False(t, hasReasoning, "Expected no reasoning part when reasoning is disabled")
				}
			}

			assert.NotEmpty(t, finalAnswer, "Final answer should not be empty")
			t.Logf("Final answer: %s", finalAnswer)

			// Verify usage includes reasoning tokens if applicable
			if hasReasoning && resp.Usage != nil {
				assert.Greater(t, resp.Usage.TotalTokens, 0, "Total tokens should be greater than 0")
				t.Logf("Token usage - Input: %d, Output: %d, Total: %d",
					resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.TotalTokens)
			}
		})
	}
}

// TestLive_GenerateWithReasoningStreaming tests the reasoning feature with streaming responses.
// Note: Currently, the Anthropic API may not stream reasoning content in real-time;
// the reasoning part might be included in the final accumulated message.
//
// To run these reasoning tests:
// ANTHROPIC_API_KEY=your_key go test -v -run TestLive_GenerateWithReasoning
func TestLive_GenerateWithReasoningStreaming(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	// Use claude-3.7-sonnet which supports reasoning
	model := anthropic.Model(g, "claude-3.7-sonnet")
	require.NotNil(t, model)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	config := struct {
		ai.GenerationCommonConfig
		anthropic.ExtendedThinkingConfig
	}{
		GenerationCommonConfig: ai.GenerationCommonConfig{
			MaxOutputTokens: 8192, // Large enough to support reasoning budget (8192 * 0.15 = 1228 > 1024, leaving 6963 for actual output)
			Temperature:     0.0,
		},
		ExtendedThinkingConfig: anthropic.ExtendedThinkingConfig{
			ExtendedThinkingEnabled:     true,
			ExtendedThinkingBudgetRatio: 0.25, // 25% of 8192 = 2048 tokens for reasoning, 6144 for actual output
		},
	}

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("What is 25% of 80? Show your reasoning step by step."),
				},
			},
		},
		Config: config,
	}

	var chunks []string
	var chunkTypes []string
	resp, err := model.Generate(ctx, req, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		if len(chunk.Content) > 0 {
			for _, part := range chunk.Content {
				if part.IsText() && part.Text != "" {
					chunks = append(chunks, part.Text)
					chunkTypes = append(chunkTypes, "text")
				} else if part.IsReasoning() && part.Text != "" {
					chunks = append(chunks, part.Text)
					chunkTypes = append(chunkTypes, "reasoning")
				}
			}
		}
		return nil
	})

	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotEmpty(t, chunks)

	// Check final response structure
	hasReasoning := false
	var reasoningText string
	var finalAnswer string

	for _, part := range resp.Message.Content {
		if part.IsReasoning() {
			hasReasoning = true
			reasoningText = part.Text
		} else if part.IsText() {
			finalAnswer = part.Text
		}
	}

	assert.True(t, hasReasoning, "Expected reasoning part in final response")
	assert.NotEmpty(t, reasoningText, "Reasoning text should not be empty")
	assert.NotEmpty(t, finalAnswer, "Final answer should not be empty")

	// Log streaming details
	t.Logf("Received %d chunks during streaming", len(chunks))
	t.Logf("Chunk types: %v", chunkTypes)
	t.Logf("Final reasoning: %s", reasoningText)
	t.Logf("Final answer: %s", finalAnswer)
}

// TestLive_Claude4AutomaticReasoning tests Claude 4's automatic reasoning capabilities.
//
// With the new defaultModelParams implementation:
// - Claude 4 models have reasoning enabled by default (Enabled: true, BudgetRatio: 0)
// - When budgetRatio is 0, it defaults to 25% of maxOutputTokens
// - Users can override this by explicitly setting ExtendedThinkingConfig
//
// This test verifies both the default automatic reasoning and explicit configuration overrides.
func TestLive_Claude4AutomaticReasoning(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	tests := []struct {
		name            string
		modelName       string
		prompt          string
		expectReasoning bool
	}{
		{
			name:            "Claude 4 - Simple calculation",
			modelName:       "claude-4-sonnet",
			prompt:          "What is 2+2?",
			expectReasoning: true, // Claude 4 has reasoning enabled by default
		},
		{
			name:            "Claude 4 - Complex problem",
			modelName:       "claude-4-sonnet",
			prompt:          "A snail climbs up a wall. During the day, it climbs 3 feet up, but at night it slides 2 feet down. If the wall is 10 feet high, how many days will it take the snail to reach the top? Think through this step by step.",
			expectReasoning: true,
		},
		{
			name:            "Claude 3.7 - Logic puzzle",
			modelName:       "claude-3.7-sonnet",
			prompt:          "Three friends (Alice, Bob, and Carol) are sitting in a row. Alice is not on the left. Bob is not on the right. Carol is not in the middle. What is the order from left to right?",
			expectReasoning: true, // Claude 3.7 also has reasoning enabled by default
		},
		{
			name:            "Claude 3.5 - Simple problem",
			modelName:       "claude-3.5-haiku",
			prompt:          "If a train travels 120 miles in 2 hours, what is its speed in mph? Think step by step.",
			expectReasoning: false, // Claude 3.5 has reasoning disabled by default
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := anthropic.Model(g, tt.modelName)
			if model == nil {
				t.Skipf("Model %s not available", tt.modelName)
			}

			// Create context with timeout
			ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
			defer cancel()

			// Test without providing ExtendedThinkingConfig at all
			// This should use the defaultModelParams
			req := &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							ai.NewTextPart(tt.prompt),
						},
					},
				},
				Config: ai.GenerationCommonConfig{
					MaxOutputTokens: 8192, // Ensure enough tokens for reasoning (8192 * 0.15 = 1228 > 1024, leaving 6963 for actual output)
					Temperature:     0.0,
				},
			}

			resp, err := model.Generate(ctx, req, nil)
			require.NoError(t, err)
			require.NotNil(t, resp)
			require.NotNil(t, resp.Message)
			require.NotEmpty(t, resp.Message.Content)

			// Check if reasoning was automatically used
			hasReasoning := false
			var reasoningText string
			var finalAnswer string

			for _, part := range resp.Message.Content {
				if part.IsReasoning() {
					hasReasoning = true
					reasoningText = part.Text
					t.Logf("Found reasoning part with %d characters", len(reasoningText))
				} else if part.IsText() {
					finalAnswer = part.Text
				}
			}

			t.Logf("Model %s - Has reasoning: %v", tt.modelName, hasReasoning)
			if hasReasoning {
				t.Logf("Reasoning text preview: %.200s...", reasoningText)
			}

			// Verify based on expected behavior from defaultModelParams
			assert.Equal(t, tt.expectReasoning, hasReasoning,
				"Model %s should have reasoning=%v based on defaultModelParams",
				tt.modelName, tt.expectReasoning)

			assert.NotEmpty(t, finalAnswer, "Final answer should not be empty")
			t.Logf("Final answer: %s", finalAnswer)

			// Verify usage includes reasoning tokens if applicable
			if hasReasoning && resp.Usage != nil {
				assert.Greater(t, resp.Usage.TotalTokens, 0, "Total tokens should be greater than 0")
				t.Logf("Token usage - Input: %d, Output: %d, Total: %d",
					resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.TotalTokens)
			}
		})
	}
}

// TestLive_Claude4ExplicitReasoningControl tests that users can override the default reasoning behavior
// by explicitly setting ExtendedThinkingConfig
func TestLive_Claude4ExplicitReasoningControl(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	tests := []struct {
		name             string
		modelName        string
		prompt           string
		extendedThinking *bool // nil means not set, true/false means explicitly set
		budgetRatio      float64
		expectReasoning  bool
	}{
		{
			name:             "Claude 4 - Explicitly disable reasoning",
			modelName:        "claude-4-sonnet",
			prompt:           "A complex math problem: If x + 2y = 10 and 3x - y = 5, what are x and y?",
			extendedThinking: gog.PtrOf(false),
			budgetRatio:      0,
			expectReasoning:  false, // Should not have reasoning when explicitly disabled
		},
		{
			name:             "Claude 3.5 - Explicitly enable reasoning (silently ignored)",
			modelName:        "claude-3.5-haiku",
			prompt:           "What is 10 * 5? Think step by step.",
			extendedThinking: gog.PtrOf(true),
			budgetRatio:      0.25,
			expectReasoning:  false, // Claude 3.5 doesn't support reasoning, API silently ignores it
		},
		{
			name:             "Claude 3.7 - Override default with disable",
			modelName:        "claude-3.7-sonnet",
			prompt:           "Solve: 2x + 3 = 7",
			extendedThinking: gog.PtrOf(false),
			budgetRatio:      0,
			expectReasoning:  false, // Should not have reasoning when explicitly disabled
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := anthropic.Model(g, tt.modelName)
			if model == nil {
				t.Skipf("Model %s not available", tt.modelName)
			}

			// Create context with timeout
			ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
			defer cancel()

			// Build config based on test case
			var config interface{}
			if tt.extendedThinking != nil {
				// Explicitly set ExtendedThinkingConfig
				config = map[string]interface{}{
					"maxOutputTokens":             2000,
					"temperature":                 0.0,
					"extendedThinkingEnabled":     *tt.extendedThinking,
					"extendedThinkingBudgetRatio": tt.budgetRatio,
				}
			} else {
				// Only set GenerationCommonConfig
				config = ai.GenerationCommonConfig{
					MaxOutputTokens: 2000,
					Temperature:     0.0,
				}
			}

			req := &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							ai.NewTextPart(tt.prompt),
						},
					},
				},
				Config: config,
			}

			resp, err := model.Generate(ctx, req, nil)
			require.NoError(t, err)
			require.NotNil(t, resp)
			require.NotNil(t, resp.Message)
			require.NotEmpty(t, resp.Message.Content)

			// Check if reasoning was used
			hasReasoning := false
			var reasoningText string
			var finalAnswer string

			for _, part := range resp.Message.Content {
				if part.IsReasoning() {
					hasReasoning = true
					reasoningText = part.Text
					t.Logf("Found reasoning part with %d characters", len(reasoningText))
				} else if part.IsText() {
					finalAnswer = part.Text
				}
			}

			t.Logf("Model %s with explicit config - Has reasoning: %v", tt.modelName, hasReasoning)
			if hasReasoning {
				t.Logf("Reasoning text preview: %.200s...", reasoningText)
			}

			// Verify based on expected behavior
			if tt.modelName == "claude-3.5-haiku" && tt.expectReasoning {
				// Claude 3.5 doesn't support reasoning, so skip assertion
				t.Logf("Note: Claude 3.5 doesn't support reasoning feature")
			} else {
				assert.Equal(t, tt.expectReasoning, hasReasoning,
					"Model %s should have reasoning=%v based on explicit config",
					tt.modelName, tt.expectReasoning)
			}

			assert.NotEmpty(t, finalAnswer, "Final answer should not be empty")
			t.Logf("Final answer: %s", finalAnswer)
		})
	}
}

func TestLive_Claud4ThinkingStreamingCompareGenerate(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	model := anthropic.Model(g, "claude-4-sonnet")
	require.NotNil(t, model)

	streamingMessage := ""
	genResp, err := genkit.Generate(
		ctx,
		g,
		ai.WithModelName("anthropic/claude-4-sonnet"),
		ai.WithPrompt("I want to order and identify my entities by Random Unique ID. But, UUID has not been ordered. So can we recommend a way to order and identify them at a time?"),
		ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
			for _, content := range chunk.Content {
				if content.IsText() {
					streamingMessage += content.Text
				}
			}
			return nil
		}),
	)
	require.NoError(t, err)
	require.NotNil(t, genResp)
	require.NotNil(t, genResp.Message)
	require.NotEmpty(t, genResp.Message.Content)

	t.Logf("streamingMessage: %s", streamingMessage)
	t.Logf("genResp: %s", genResp.Text())

	assert.Equal(t, streamingMessage, genResp.Text())
}

// TestLive_GenerateWithToolCallStreaming tests tool calling with streaming to verify
// InputJSONDelta handling in the streaming response processing
func TestLive_GenerateWithToolCallStreaming(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	model := anthropic.Model(g, "claude-3.5-haiku")
	require.NotNil(t, model)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	// Define a simple tool for testing
	tool := &ai.ToolDefinition{
		Name:        "get_weather",
		Description: "Get the current weather for a location",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "The city and state, e.g. San Francisco, CA",
				},
				"unit": map[string]any{
					"type":        "string",
					"enum":        []string{"celsius", "fahrenheit"},
					"description": "The unit of temperature",
				},
			},
			"required": []string{"location"},
		},
	}

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("What's the weather like in Tokyo, Japan? Use celsius for temperature."),
				},
			},
		},
		Config: &ai.GenerationCommonConfig{
			MaxOutputTokens: 1000,
			Temperature:     0.0,
		},
		Tools: []*ai.ToolDefinition{tool},
	}

	var streamedChunks []*ai.ModelResponseChunk
	var toolRequestChunks []*ai.Part

	resp, err := model.Generate(ctx, req, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		streamedChunks = append(streamedChunks, chunk)

		// Collect tool request parts specifically to test InputJSONDelta handling
		for _, part := range chunk.Content {
			if part.IsToolRequest() {
				toolRequestChunks = append(toolRequestChunks, part)
				inputBytes, _ := json.Marshal(part.ToolRequest.Input)
				fmt.Printf("Tool request chunk: Name=%s, Input=%s\n",
					part.ToolRequest.Name, string(inputBytes))
			}
		}

		return nil
	})

	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotEmpty(t, streamedChunks)

	// Check that we received tool request chunks during streaming
	assert.NotEmpty(t, toolRequestChunks, "Expected to receive tool request chunks during streaming")

	// Verify the final response contains tool calls
	hasToolCall := false
	for _, part := range resp.Message.Content {
		if part.IsToolRequest() {
			hasToolCall = true
			assert.Equal(t, "get_weather", part.ToolRequest.Name)
			assert.NotEmpty(t, part.ToolRequest.Input)

			// Verify the tool input contains expected fields
			var input map[string]any
			inputBytes, err := json.Marshal(part.ToolRequest.Input)
			require.NoError(t, err)
			err = json.Unmarshal(inputBytes, &input)
			require.NoError(t, err)
			assert.Contains(t, input, "location")

			t.Logf("Final tool call: %+v", part.ToolRequest)
		}
	}

	assert.True(t, hasToolCall, "Expected response to contain tool calls")
	t.Logf("Total streamed chunks: %d", len(streamedChunks))
	t.Logf("Tool request chunks: %d", len(toolRequestChunks))
}

// TestLive_GenerateWithComplexToolCallStreaming tests more complex tool calling scenarios
// to ensure InputJSONDelta handling works with larger JSON inputs
func TestLive_GenerateWithComplexToolCallStreaming(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	model := anthropic.Model(g, "claude-3.5-haiku")
	require.NotNil(t, model)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	// Define a tool with complex input schema
	tool := &ai.ToolDefinition{
		Name:        "create_calendar_event",
		Description: "Create a calendar event with detailed information",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"title": map[string]any{
					"type":        "string",
					"description": "Event title",
				},
				"description": map[string]any{
					"type":        "string",
					"description": "Event description",
				},
				"start_time": map[string]any{
					"type":        "string",
					"description": "Start time in ISO format",
				},
				"end_time": map[string]any{
					"type":        "string",
					"description": "End time in ISO format",
				},
				"attendees": map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type": "string",
							},
							"email": map[string]any{
								"type": "string",
							},
						},
					},
				},
				"location": map[string]any{
					"type":        "string",
					"description": "Meeting location",
				},
			},
			"required": []string{"title", "start_time", "end_time"},
		},
	}

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("Create a calendar event for a team meeting tomorrow at 2 PM for 1 hour. Title should be 'Weekly Team Sync'. Include attendees Alice (alice@example.com) and Bob (bob@example.com). Location should be 'Conference Room A'."),
				},
			},
		},
		Config: &ai.GenerationCommonConfig{
			MaxOutputTokens: 1500,
			Temperature:     0.0,
		},
		Tools: []*ai.ToolDefinition{tool},
	}

	var jsonDeltaChunks []string
	var finalToolInput map[string]any

	resp, err := model.Generate(ctx, req, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		// Track partial JSON inputs to test InputJSONDelta handling
		for _, part := range chunk.Content {
			if part.IsToolRequest() {
				inputBytes, _ := json.Marshal(part.ToolRequest.Input)
				jsonDeltaChunks = append(jsonDeltaChunks, string(inputBytes))
			}
		}

		return nil
	})

	require.NoError(t, err)
	require.NotNil(t, resp)

	// Verify the final response contains the expected tool call
	hasToolCall := false
	for _, part := range resp.Message.Content {
		if part.IsToolRequest() {
			hasToolCall = true
			assert.Equal(t, "create_calendar_event", part.ToolRequest.Name)

			// Parse and verify the final JSON input
			inputBytes, err := json.Marshal(part.ToolRequest.Input)
			require.NoError(t, err)
			err = json.Unmarshal(inputBytes, &finalToolInput)
			require.NoError(t, err)

			// Verify required fields are present
			assert.Contains(t, finalToolInput, "title")
			assert.Contains(t, finalToolInput, "start_time")
			assert.Contains(t, finalToolInput, "end_time")

			t.Logf("Final tool input: %+v", finalToolInput)
		}
	}

	assert.True(t, hasToolCall, "Expected response to contain tool calls")

	// Log streaming behavior for debugging
	t.Logf("Total JSON delta chunks received: %d", len(jsonDeltaChunks))
	if len(jsonDeltaChunks) > 0 {
		t.Logf("First chunk: %s", jsonDeltaChunks[0])
		t.Logf("Last chunk: %s", jsonDeltaChunks[len(jsonDeltaChunks)-1])
	}
}

// TestLive_GenerateWithMultipleToolCallsStreaming tests scenarios with multiple tool calls
// to ensure InputJSONDelta handling works correctly with tool index matching
func TestLive_GenerateWithMultipleToolCallsStreaming(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	model := anthropic.Model(g, "claude-3.5-haiku")
	require.NotNil(t, model)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	// Define multiple tools
	weatherTool := &ai.ToolDefinition{
		Name:        "get_weather",
		Description: "Get current weather",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{
					"type": "string",
				},
			},
			"required": []string{"location"},
		},
	}

	timeTool := &ai.ToolDefinition{
		Name:        "get_time",
		Description: "Get current time",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"timezone": map[string]any{
					"type": "string",
				},
			},
			"required": []string{"timezone"},
		},
	}

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("What's the weather in Tokyo and what time is it there?"),
				},
			},
		},
		Config: &ai.GenerationCommonConfig{
			MaxOutputTokens: 1500,
			Temperature:     0.0,
		},
		Tools: []*ai.ToolDefinition{weatherTool, timeTool},
	}

	var toolCallsByIndex = make(map[int][]*ai.Part)

	resp, err := model.Generate(ctx, req, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		// Track tool calls by index to verify InputJSONDelta index matching
		for _, part := range chunk.Content {
			if part.IsToolRequest() {
				// For this test, we'll track tool calls but can't easily get the index
				// from the chunk directly, so we'll just verify they exist
				toolCallsByIndex[0] = append(toolCallsByIndex[0], part)
			}
		}

		return nil
	})

	require.NoError(t, err)
	require.NotNil(t, resp)

	// Count tool calls in final response
	toolCallCount := 0
	for _, part := range resp.Message.Content {
		if part.IsToolRequest() {
			toolCallCount++
			inputBytes, _ := json.Marshal(part.ToolRequest.Input)
			t.Logf("Tool call: %s with input: %s",
				part.ToolRequest.Name, string(inputBytes))
		}
	}

	// We expect at least one tool call, possibly two depending on the model's response
	assert.GreaterOrEqual(t, toolCallCount, 1, "Expected at least one tool call")
	assert.NotEmpty(t, toolCallsByIndex[0], "Expected to receive tool request chunks during streaming")

	t.Logf("Total tool calls in final response: %d", toolCallCount)
	t.Logf("Tool request chunks during streaming: %d", len(toolCallsByIndex[0]))
}

func TestLive_GenerateWithWebSearchStreaming(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	model := anthropic.Model(g, "claude-3.5-haiku")
	require.NotNil(t, model)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	// Define web search tool
	webSearchTool := &ai.ToolDefinition{
		Name:        "web_search",
		Description: "Search the web for current information",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "The search query",
				},
			},
			"required": []string{"query"},
		},
	}

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("What's the latest news about AI in 2025? Please search for recent information."),
				},
			},
		},
		Config: &ai.GenerationCommonConfig{
			MaxOutputTokens: 2000,
			Temperature:     0.0,
		},
		Tools: []*ai.ToolDefinition{webSearchTool},
	}

	var streamedChunks []*ai.ModelResponseChunk
	var toolRequests []*ai.ToolRequest
	var toolResponses []*ai.ToolResponse
	var citationBlocks []map[string]any

	resp, err := model.Generate(ctx, req, func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		streamedChunks = append(streamedChunks, chunk)

		if len(chunk.Content) > 0 {
			for _, part := range chunk.Content {
				// Collect ToolRequest data (web search initiation)
				if part.IsToolRequest() {
					toolRequests = append(toolRequests, part.ToolRequest)
				}
				// Collect ToolResponse data (web search completion)
				if part.IsToolResponse() {
					toolResponses = append(toolResponses, part.ToolResponse)
				}
				// Collect Citation data (actual web search results)
				if part.IsCustom() {
					if customType, ok := part.Custom["type"].(string); ok && customType == "citation" {
						citationBlocks = append(citationBlocks, part.Custom)
					}
				}
			}
		}
		return nil
	})

	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotEmpty(t, streamedChunks)

	// Assert web search functionality - these MUST work
	assert.NotEmpty(t, toolRequests, "Web search ToolRequest must be detected")
	assert.NotEmpty(t, toolResponses, "Web search ToolResponse must be received")
	assert.NotEmpty(t, citationBlocks, "Citation blocks with search results must be received")

	// Verify ToolRequest is for web_search
	webSearchRequestFound := false
	for _, toolReq := range toolRequests {
		if toolReq.Name == "web_search" {
			webSearchRequestFound = true
			assert.NotEmpty(t, toolReq.Ref, "Web search ToolRequest must have a reference ID")
			break
		}
	}
	assert.True(t, webSearchRequestFound, "Web search ToolRequest must be found")

	// Verify ToolResponse is from web_search
	webSearchResponseFound := false
	var webSearchOutput string
	for _, toolResp := range toolResponses {
		if toolResp.Name == "web_search" {
			webSearchResponseFound = true
			assert.NotEmpty(t, toolResp.Ref, "Web search ToolResponse must have a reference ID")
			assert.NotNil(t, toolResp.Output, "Web search ToolResponse must have output data")

			// Extract and verify the actual search results from ToolResponse
			if toolResp.Output != nil {
				webSearchOutput = string(toolResp.Output.(json.RawMessage))
				assert.NotEmpty(t, webSearchOutput, "Web search ToolResponse output must not be empty")

				// Log the actual web search results from ToolResponse
				outputPreview := webSearchOutput
				if len(outputPreview) > 500 {
					outputPreview = outputPreview[:500] + "..."
				}
				t.Logf("DEBUG: ToolResponse web search output: %s", outputPreview)

				// Verify it contains search result structure
				assert.True(t,
					strings.Contains(webSearchOutput, "results") ||
						strings.Contains(webSearchOutput, "content") ||
						strings.Contains(webSearchOutput, "title"),
					"ToolResponse output should contain search result data")
			}
			break
		}
	}
	assert.True(t, webSearchResponseFound, "Web search ToolResponse must be found")

	// Verify citation blocks contain web search results
	validCitationsFound := 0
	for _, citation := range citationBlocks {
		if body, ok := citation["body"].(map[string]any); ok {
			// Check for web search result structure
			if searchType, hasType := body["type"].(string); hasType && searchType == "web_search_result_location" {
				validCitationsFound++

				// Verify citation has required fields
				assert.NotEmpty(t, body["title"], "Citation must have title")
				assert.NotEmpty(t, body["url"], "Citation must have URL")
				assert.NotEmpty(t, body["cited_text"], "Citation must have cited text")
			}
		}
	}
	assert.Greater(t, validCitationsFound, 0, "Must have at least one valid web search citation")

	// Verify final response contains AI-generated content based on search results
	hasTextContent := false
	totalTextLength := 0
	for _, part := range resp.Message.Content {
		if part.IsText() && part.Text != "" {
			hasTextContent = true
			totalTextLength += len(part.Text)
		}
	}
	assert.True(t, hasTextContent, "Response must contain text content")
	assert.Greater(t, totalTextLength, 100, "Response must contain substantial content (>100 chars)")

	// Log summary for debugging
	t.Logf("✅ Web search test passed:")
	t.Logf("  - Streamed chunks: %d", len(streamedChunks))
	t.Logf("  - ToolRequest chunks: %d", len(toolRequests))
	t.Logf("  - ToolResponse chunks: %d", len(toolResponses))
	t.Logf("  - Citation blocks: %d", len(citationBlocks))
	t.Logf("  - Valid search citations: %d", validCitationsFound)
	t.Logf("  - Web search output length: %d chars", len(webSearchOutput))
	t.Logf("  - Final response length: %d chars", totalTextLength)
}

func TestLive_GenerateWithPDF(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}))

	model := anthropic.Model(g, "claude-3.5-haiku")
	require.NotNil(t, model)

	// Download PDF from the given URL
	pdfURL := "https://arxiv.org/pdf/2109.10282"
	resp, err := http.Get(pdfURL)
	require.NoError(t, err, "Failed to download PDF")
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode, "Failed to download PDF: status %d", resp.StatusCode)

	// Read PDF content
	pdfData, err := io.ReadAll(resp.Body)
	require.NoError(t, err, "Failed to read PDF data")
	require.NotEmpty(t, pdfData, "PDF data should not be empty")

	tests := []struct {
		name     string
		question string
		timeout  time.Duration
	}{
		{
			name:     "TrOCR main concept",
			question: "What is TrOCR and what makes it different from traditional OCR approaches?",
			timeout:  2 * time.Minute,
		},
		{
			name:     "TrOCR architecture",
			question: "Describe the architecture of TrOCR. What are the main components?",
			timeout:  2 * time.Minute,
		},
		{
			name:     "TrOCR advantages",
			question: "What are the three main advantages of TrOCR mentioned in the paper?",
			timeout:  2 * time.Minute,
		},
		{
			name:     "TrOCR evaluation",
			question: "On which types of text recognition tasks was TrOCR evaluated?",
			timeout:  2 * time.Minute,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create context with timeout
			ctx, cancel := context.WithTimeout(context.Background(), tt.timeout)
			defer cancel()

			req := &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							ai.NewTextPart(tt.question),
							ai.NewMediaPart("application/pdf", "https://arxiv.org/pdf/2109.10282"),
						},
					},
				},
			}

			resp, err := model.Generate(ctx, req, nil)
			require.NoError(t, err)
			require.NotNil(t, resp)
			require.NotNil(t, resp.Message)
			require.NotEmpty(t, resp.Message.Content)

			content := resp.Message.Content[0].Text
			assert.NotEmpty(t, content)

			// Basic validation that the response is relevant to TrOCR
			contentLower := strings.ToLower(content)
			assert.True(t,
				strings.Contains(contentLower, "trocr") ||
					strings.Contains(contentLower, "transformer") ||
					strings.Contains(contentLower, "ocr"),
				"Response should mention TrOCR, transformer, or OCR concepts")

			t.Logf("Q: %s\nA: %s\n\n", tt.question, content)
		})
	}
}
