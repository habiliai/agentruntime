package engine_test

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/habiliai/agentruntime/config"
	"github.com/habiliai/agentruntime/engine"
	"github.com/habiliai/agentruntime/entity"
	genkitinternal "github.com/habiliai/agentruntime/internal/genkit"
	"github.com/habiliai/agentruntime/knowledge"
	"github.com/habiliai/agentruntime/memory"
	"github.com/habiliai/agentruntime/tool"
)

// TestMissmatchStreamingAndOutput demonstrates and tests for potential mismatches
// between streaming responses and final output when tool calls are involved.
//
// This test creates a scenario where an agent uses tools (weather tool) and
// compares:
// - Content received through streaming callback (incremental chunks)
// - Final output content and structured tool calls
//
// The test helps identify issues where:
// - Streaming shows partial/incomplete tool information
// - Final output has properly structured tool calls
// - Content lengths or formats differ between streaming and final output
//
// To run this test, set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables.
func (s *EngineTestSuite) TestMissmatchStreamingAndOutput() {
	// Skip this test if no API key is provided
	if os.Getenv("OPENAI_API_KEY") == "" && os.Getenv("ANTHROPIC_API_KEY") == "" {
		s.T().Skip("No API key provided, skipping test")
		return
	}

	ctx := s.T().Context()

	// Create a tool manager with a simple weather tool
	skills := []entity.AgentSkillUnion{
		{
			Type: "nativeTool",
			OfNative: &entity.NativeAgentSkill{
				Name:    "get_weather",
				Details: "Get weather information when you need it",
				Env: map[string]any{
					"OPENWEATHER_API_KEY": os.Getenv("OPENWEATHER_API_KEY"),
				},
			},
		},
	}

	g := genkitinternal.NewGenkit(s, &config.ModelConfig{
		OpenAIAPIKey:    os.Getenv("OPENAI_API_KEY"),
		AnthropicAPIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}, slog.Default(), true)

	memoryService, err := memory.NewService(s, &config.ModelConfig{
		OpenAIAPIKey:    os.Getenv("OPENAI_API_KEY"),
		AnthropicAPIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}, &config.MemoryConfig{
		GenerationModel: "openai/gpt-4.1-mini",
	}, slog.Default())
	s.Require().NoError(err)

	knowledgeService, err := knowledge.NewService(s, &config.ModelConfig{
		OpenAIAPIKey:    os.Getenv("OPENAI_API_KEY"),
		AnthropicAPIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}, &config.KnowledgeConfig{
		SqliteEnabled: true,
		SqlitePath:    ":memory:",
		VectorEnabled: true,
	}, slog.Default())
	s.Require().NoError(err)

	toolManager, err := tool.NewToolManager(ctx, skills, slog.Default(), g, knowledgeService, memoryService)
	s.Require().NoError(err)
	defer toolManager.Close()

	// Create engine with tool manager
	testEngine := engine.NewEngine(
		slog.Default(),
		toolManager,
		g,
	)

	// Create an agent that will use tools
	agent := entity.Agent{
		Name: "WeatherBot",
		Role: "weather assistant",
		System: fmt.Sprintf(`
		<today>
		Today is %s
		</today>
		`, time.Now().Format("2006-01-02")),
		Prompt:    "You are a weather assistant. When users ask about weather, use the get_weather tool to provide accurate information.",
		ModelName: "anthropic/claude-4-sonnet",
		Skills: []entity.AgentSkillUnion{
			{
				Type: "nativeTool",
				OfNative: &entity.NativeAgentSkill{
					Name:    "get_weather",
					Details: "Get weather information when you need it",
					Env: map[string]any{
						"OPENWEATHER_API_KEY": os.Getenv("OPENWEATHER_API_KEY"),
					},
				},
			},
			{
				Type: "mcp",
				OfMCP: &entity.MCPAgentSkill{
					Command: "uvx",
					Args:    []string{"mcp-server-time"},
				},
			},
		},
	}

	// Create a request that will trigger a tool call
	req := engine.RunRequest{
		ThreadInstruction: "User is asking about weather information",
		History: []engine.Conversation{
			{
				User: "USER",
				Text: "What's the weather like in Tokyo today?",
			},
		},
		Participant: []engine.Participant{
			{
				Name:        "WeatherBot",
				Description: "Weather assistant",
				Role:        "assistant",
			},
		},
	}

	// Capture streaming output
	var streamingContent strings.Builder
	var streamingChunks []string

	streamCallback := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		if chunk.Content != nil {
			for _, part := range chunk.Content {
				if part.Text != "" {
					streamingContent.WriteString(part.Text)
					streamingChunks = append(streamingChunks, part.Text)
				}
			}
		}
		return nil
	}

	// Run the engine with streaming
	result, err := testEngine.Run(ctx, agent, req, streamCallback)
	s.Require().NoError(err)
	s.Require().NotNil(result)

	// Get the final output text
	finalText := ""
	if result.ModelResponse != nil && result.ModelResponse.Message != nil {
		for _, part := range result.ModelResponse.Message.Content {
			if part.Text != "" {
				finalText += part.Text
			}
		}
	}

	streamingText := streamingContent.String()

	// Log the results for debugging
	s.T().Logf("=== STREAMING OUTPUT ===")
	s.T().Logf("Total streaming chunks: %d", len(streamingChunks))
	s.T().Logf("Streaming content: %s", streamingText)

	s.T().Logf("=== FINAL OUTPUT ===")
	s.T().Logf("Final content: %s", finalText)
	s.T().Logf("Tool calls count: %d", len(result.ToolCalls))
	for i, toolCall := range result.ToolCalls {
		s.T().Logf("Tool call %d: %s", i, toolCall.Name)
		s.T().Logf("  Arguments: %s", string(toolCall.Arguments))
		s.T().Logf("  Result: %s", string(toolCall.Result))
	}

	// Check if streaming content contains tool call information
	hasToolInfoInStream := strings.Contains(streamingText, "get_weather") ||
		strings.Contains(streamingText, "tool_call") ||
		strings.Contains(streamingText, "function_call")

	// Check if final output properly separates tool calls
	hasProperToolCalls := len(result.ToolCalls) > 0

	if hasToolInfoInStream && hasProperToolCalls {
		s.T().Logf("✓ MISMATCH DETECTED: Tool information appears in streaming but is structured differently in final output")
		s.T().Logf("  - Streaming contains tool-related content: %v", hasToolInfoInStream)
		s.T().Logf("  - Final output has structured tool calls: %v", hasProperToolCalls)
	}

	// Additional checks for common mismatch patterns
	if len(streamingChunks) > 0 && len(result.ToolCalls) > 0 {
		s.T().Logf("✓ POTENTIAL MISMATCH: Streaming delivered %d chunks but final output has %d structured tool calls",
			len(streamingChunks), len(result.ToolCalls))

		// Check if the content structure is different
		if streamingText != finalText {
			s.T().Logf("✓ CONTENT MISMATCH: Streaming text differs from final text")
			s.T().Logf("  Streaming length: %d", len(streamingText))
			s.T().Logf("  Final length: %d", len(finalText))
		}
	}

	// This test demonstrates the issue rather than asserting a specific condition
	// The mismatch is expected behavior that we want to document and potentially fix
	fmt.Printf("\n=== MISMATCH ANALYSIS ===\n")
	fmt.Printf("Streaming chunks: %d\n", len(streamingChunks))
	fmt.Printf("Final tool calls: %d\n", len(result.ToolCalls))
	fmt.Printf("Streaming content length: %d\n", len(streamingText))
	fmt.Printf("Final content length: %d\n", len(finalText))
	fmt.Printf("Content differs: %v\n", streamingText != finalText)
}
