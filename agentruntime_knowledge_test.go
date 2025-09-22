package agentruntime_test

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"os"
	"strings"
	"testing"

	"github.com/goccy/go-yaml"
	"github.com/habiliai/agentruntime"
	"github.com/habiliai/agentruntime/config"
	"github.com/habiliai/agentruntime/engine"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/knowledge"
	_ "github.com/joho/godotenv/autoload"
	"github.com/stretchr/testify/require"
)

func TestAgentWithKnowledgeService(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("Skipping test because OPENAI_API_KEY is not set")
	}
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("Skipping test because ANTHROPIC_API_KEY is not set")
	}

	bytes, err := os.ReadFile("examples/example_knowledge.agent.yaml")
	require.NoError(t, err)

	var agent entity.Agent
	err = yaml.Unmarshal(bytes, &agent)
	require.NoError(t, err)

	// Test basic agent information
	require.Equal(t, "HosuAgent", agent.Name, "Agent name should be 'HosuAgent'")
	require.Contains(t, agent.Description, "rescue dogs", "Agent description should contain rescue dogs info")
	require.Equal(t, "anthropic/claude-3.5-haiku", agent.ModelName, "Model name should be 'anthropic/claude-3.5-haiku'")
	require.Equal(t, "Rescue Dog Knowledge Assistant", agent.Role, "Role should be 'Rescue Dog Knowledge Assistant'")

	// Test system and prompt
	require.Contains(t, agent.System, "rescue dogs and animal welfare", "System should mention rescue dogs")
	require.Contains(t, agent.Prompt, "HosuAgent", "Prompt should contain agent name")
	require.Contains(t, agent.Prompt, "pet adoption", "Prompt should mention pet adoption")

	// Test knowledge entries
	require.Len(t, agent.Knowledge, 4, "Should have 4 knowledge entries")

	// Test first knowledge entry (Hosu info)
	hosuKnowledge := agent.Knowledge[0]
	require.Equal(t, "Hosu", hosuKnowledge["dogName"], "First knowledge should be about Hosu")
	require.Equal(t, "Mandu, Hoshu, Hodol", hosuKnowledge["aliases"], "Hosu aliases should match")
	require.Equal(t, "Mixed breed", hosuKnowledge["breed"], "Hosu breed should be Mixed breed")
	require.Equal(t, "3 years", hosuKnowledge["age"], "Hosu age should be 3 years")

	// Test second knowledge entry (Nuri shelter)
	shelterKnowledge := agent.Knowledge[1]
	require.Equal(t, "Nuri", shelterKnowledge["hometown"], "Second knowledge should be about Nuri shelter")
	require.Equal(t, "Seoul, South Korea", shelterKnowledge["location"], "Shelter location should be Seoul")
	require.Equal(t, "50 dogs", shelterKnowledge["capacity"], "Shelter capacity should be 50 dogs")

	// Test message examples
	require.Len(t, agent.MessageExamples, 2, "Should have 2 message examples")
	firstExample := agent.MessageExamples[0]
	require.Contains(t, firstExample[0].Text, "Tell me about Hosu", "First example should ask about Hosu")
	// First example should have knowledge_search action since knowledge is accessed via tool
	require.Contains(t, firstExample[1].Actions, "knowledge_search", "First example should use knowledge_search")

	// Second example should have web_search action
	secondExample := agent.MessageExamples[1]
	require.Contains(t, secondExample[0].Text, "adopting a rescue dog", "Second example should ask about adoption")
	require.Contains(t, secondExample[1].Actions, "web_search", "Second example should use web_search")

	// Test skills - now should include knowledge_search skill
	require.GreaterOrEqual(t, len(agent.Skills), 3, "Should have at least 3 skills")

	// Find and test the skills
	var webSearchSkill *entity.NativeAgentSkill
	var adoptionAdvisorSkill *entity.LLMAgentSkill
	var knowledgeSearchSkill *entity.NativeAgentSkill
	for i, skill := range agent.Skills {
		switch skill.Type {
		case "nativeTool":
			switch skill.OfNative.Name {
			case "web_search":
				webSearchSkill = agent.Skills[i].OfNative
			case "knowledge_search":
				knowledgeSearchSkill = agent.Skills[i].OfNative
			}
		case "llm":
			switch skill.OfLLM.Name {
			case "adoption_advisor":
				adoptionAdvisorSkill = agent.Skills[i].OfLLM
			}
		}
	}
	require.NotNil(t, webSearchSkill, "Should have web_search skill")

	require.NotNil(t, adoptionAdvisorSkill, "Should have adoption_advisor skill")
	require.Contains(t, adoptionAdvisorSkill.Description, "adoption and care", "adoption_advisor description should mention adoption and care")

	require.NotNil(t, knowledgeSearchSkill, "Should have knowledge_search skill")

	// Test metadata
	require.NotNil(t, agent.Metadata, "Metadata should not be nil")
	require.Equal(t, "1.0", agent.Metadata["version"], "Version should be 1.0")
	require.Equal(t, "Rescue dogs and animal welfare", agent.Metadata["specialization"], "Specialization should match")

	knowledgeService, err := knowledge.NewService(context.TODO(), &config.ModelConfig{
		OpenAIAPIKey:    os.Getenv("OPENAI_API_KEY"),
		AnthropicAPIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}, config.NewKnowledgeConfig(), slog.Default())
	require.NoError(t, err)
	defer knowledgeService.Close()

	// Test runtime creation and execution with knowledge query
	runtime, err := agentruntime.NewAgentRuntime(
		context.TODO(),
		agentruntime.WithAgent(agent),
		agentruntime.WithOpenAIAPIKey(os.Getenv("OPENAI_API_KEY")),
		agentruntime.WithAnthropicAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		agentruntime.WithKnowledgeService(knowledgeService),
		agentruntime.WithLogger(slog.Default()),
	)
	require.NoError(t, err)
	defer runtime.Close()

	var out string
	resp, err := runtime.Run(context.TODO(), engine.RunRequest{
		ThreadInstruction: "User asks about a rescue dog.",
		History: []engine.Conversation{
			{
				User: "USER",
				Text: "Can you tell me about Hosu? I heard he's also called Mandu sometimes.",
			},
		},
	}, nil)
	require.NoError(t, err)
	require.NotNil(t, resp)

	out = resp.Text()
	t.Logf("Response: %+v", resp)
	t.Logf("Output: %s", out)

	// Verify the output mentions Hosu
	require.NotEmpty(t, out, "Output should not be empty")
	// The agent should be able to reference the knowledge about Hosu
	// Check for any mention of the dog or related terms
	outputLower := strings.ToLower(out)
	hasRelevantContent := strings.Contains(outputLower, "hosu") ||
		strings.Contains(outputLower, "mandu") ||
		strings.Contains(outputLower, "rescue") ||
		strings.Contains(outputLower, "nuri")
	require.True(t, hasRelevantContent,
		"Output should mention Hosu, Mandu, rescue, or Nuri context")

	// Check if knowledge_search tool was called
	knowledgeSearchCalled := false
	for _, toolCall := range resp.ToolCalls {
		if toolCall.Name == "knowledge_search" {
			knowledgeSearchCalled = true
			t.Logf("Knowledge search tool was called with arguments: %s", string(toolCall.Arguments))
			t.Logf("Knowledge search results: %s", string(toolCall.Result))

			// Verify the tool call arguments contain relevant query
			require.Contains(t, strings.ToLower(string(toolCall.Arguments)), "hosu",
				"Knowledge search query should contain 'hosu'")
		}
	}

	// With the new tool-based approach, knowledge_search should be called
	require.True(t, knowledgeSearchCalled, "knowledge_search tool should have been called")

	// Additional verification for tool-based knowledge retrieval
	// Check if the response contains specific details from the knowledge base
	detailsFound := strings.Contains(outputLower, "3 years") ||
		strings.Contains(outputLower, "mixed breed") ||
		strings.Contains(outputLower, "gentle") ||
		strings.Contains(outputLower, "playful") ||
		strings.Contains(outputLower, "belly rubs") ||
		strings.Contains(outputLower, "seoul")

	t.Logf("Knowledge details found: %v", detailsFound)

	// Log a message about the new tool-based approach
	if knowledgeSearchCalled && detailsFound {
		t.Log("Tool-based knowledge retrieval is working correctly - agent called knowledge_search and used the results")
	} else if knowledgeSearchCalled && !detailsFound {
		t.Log("Warning: Agent called knowledge_search but may not have used the results effectively")
	} else {
		t.Log("Warning: Agent did not call knowledge_search tool - may need to update agent configuration")
	}
}

// TestAgentWithRAGAndCustomKnowledge tests RAG with more complex queries
func TestAgentWithRAGAndCustomKnowledge(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("Skipping test because OPENAI_API_KEY is not set")
	}
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("Skipping test because ANTHROPIC_API_KEY is not set")
	}
	if os.Getenv("NOMIC_API_KEY") == "" {
		t.Skip("Skipping test because NOMIC_API_KEY is not set (required for embeddings)")
	}

	// Create a simple agent with knowledge
	agent := entity.Agent{
		Name:        "TestRAGAgent",
		Description: "A test agent for RAG functionality",
		ModelName:   "anthropic/claude-3.5-haiku",
		System:      "You are a helpful assistant with access to a knowledge base.",
		Role:        "Knowledge Assistant",
		Knowledge: []map[string]any{
			{
				"topic":   "Company Policy",
				"content": "Our company vacation policy allows 15 days of paid time off per year.",
				"details": "Vacation days must be approved by your manager at least 2 weeks in advance.",
			},
			{
				"topic":   "Office Hours",
				"content": "Office hours are Monday to Friday, 9 AM to 6 PM.",
				"details": "Remote work is allowed on Wednesdays and Fridays.",
			},
			{
				"topic":   "Health Benefits",
				"content": "Full health insurance coverage including dental and vision.",
				"details": "Coverage begins on the first day of employment. Family members can be added.",
			},
		},
		// Add knowledge_search skill for tool-based knowledge retrieval
		Skills: []entity.AgentSkillUnion{
			{
				Type: "nativeTool",
				OfNative: &entity.NativeAgentSkill{
					Name:    "knowledge_search",
					Details: "Search through the knowledge base for relevant information",
				},
			},
		},
	}

	// Create knowledge config with specific settings
	knowledgeConfig := config.NewKnowledgeConfig()
	knowledgeConfig.SqliteEnabled = true
	knowledgeConfig.SqlitePath = ":memory:"
	knowledgeConfig.RerankEnabled = true
	knowledgeConfig.RerankTopK = 2
	knowledgeConfig.VectorEnabled = true

	knowledgeService, err := knowledge.NewService(context.TODO(), &config.ModelConfig{
		OpenAIAPIKey:    os.Getenv("OPENAI_API_KEY"),
		AnthropicAPIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}, knowledgeConfig, slog.Default())
	require.NoError(t, err)
	defer knowledgeService.Close()

	runtime, err := agentruntime.NewAgentRuntime(
		context.TODO(),
		agentruntime.WithAgent(agent),
		agentruntime.WithOpenAIAPIKey(os.Getenv("OPENAI_API_KEY")),
		agentruntime.WithAnthropicAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		agentruntime.WithKnowledgeService(knowledgeService),
		agentruntime.WithLogger(slog.Default()),
	)
	require.NoError(t, err)
	defer runtime.Close()

	// Test 1: Query about vacation policy
	var out1 string
	resp1, err := runtime.Run(context.TODO(), engine.RunRequest{
		History: []engine.Conversation{
			{
				User: "USER",
				Text: "How many vacation days do I get?",
			},
		},
	}, nil)
	require.NoError(t, err)
	require.NotNil(t, resp1)

	out1 = resp1.Text()
	t.Logf("Test 1 - Vacation Query Response: %s", out1)
	outputLower1 := strings.ToLower(out1)
	has15Days := strings.Contains(outputLower1, "15 days") ||
		(strings.Contains(outputLower1, "15") && (strings.Contains(outputLower1, "vacation days") || strings.Contains(outputLower1, "days")))
	require.True(t, has15Days, "Should mention 15 days of vacation")

	// Verify knowledge_search tool was called for vacation query
	vacationSearchCalled := false
	for _, toolCall := range resp1.ToolCalls {
		if toolCall.Name == "knowledge_search" {
			vacationSearchCalled = true
			t.Logf("Test 1 - Knowledge search called with: %s", string(toolCall.Arguments))
		}
	}
	require.True(t, vacationSearchCalled, "knowledge_search should be called for vacation query")

	// Test 2: Query about remote work
	var out2 string
	resp2, err := runtime.Run(context.TODO(), engine.RunRequest{
		History: []engine.Conversation{
			{
				User: "USER",
				Text: "Can I work from home?",
			},
		},
	}, nil)
	require.NoError(t, err)
	require.NotNil(t, resp2)

	out2 = resp2.Text()
	t.Logf("Test 2 - Remote Work Query Response: %s", out2)
	outputLower := strings.ToLower(out2)
	hasRemoteInfo := strings.Contains(outputLower, "wednesday") ||
		strings.Contains(outputLower, "friday") ||
		strings.Contains(outputLower, "remote")
	require.True(t, hasRemoteInfo, "Should mention remote work days")

	// Verify knowledge_search tool was called for remote work query
	remoteSearchCalled := false
	for _, toolCall := range resp2.ToolCalls {
		if toolCall.Name == "knowledge_search" {
			remoteSearchCalled = true
			t.Logf("Test 2 - Knowledge search called with: %s", string(toolCall.Arguments))
		}
	}
	require.True(t, remoteSearchCalled, "knowledge_search should be called for remote work query")

	// Test 3: Query about health benefits
	var out3 string
	resp3, err := runtime.Run(context.TODO(), engine.RunRequest{
		History: []engine.Conversation{
			{
				User: "USER",
				Text: "What health benefits are included?",
			},
		},
	}, nil)
	require.NoError(t, err)
	require.NotNil(t, resp3)

	out3 = resp3.Text()
	t.Logf("Test 3 - Health Benefits Query Response: %s", out3)
	outputLower3 := strings.ToLower(out3)
	hasHealthInfo := strings.Contains(outputLower3, "health insurance") ||
		strings.Contains(outputLower3, "dental") ||
		strings.Contains(outputLower3, "vision")
	require.True(t, hasHealthInfo, "Should mention health insurance details")

	// Verify knowledge_search tool was called for health benefits query
	healthSearchCalled := false
	for _, toolCall := range resp3.ToolCalls {
		if toolCall.Name == "knowledge_search" {
			healthSearchCalled = true
			t.Logf("Test 3 - Knowledge search called with: %s", string(toolCall.Arguments))
		}
	}
	require.True(t, healthSearchCalled, "knowledge_search should be called for health benefits query")

	t.Log("All tests passed - Tool-based knowledge retrieval is working correctly")
}

func TestAgentWithRAGAndPDFKnowledge(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("Skipping test because OPENAI_API_KEY is not set")
	}
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("Skipping test because ANTHROPIC_API_KEY is not set")
	}
	if os.Getenv("NOMIC_API_KEY") == "" {
		t.Skip("Skipping test because NOMIC_API_KEY is not set (required for embeddings)")
	}

	ctx := context.TODO()
	pdfFile, err := os.ReadFile("./knowledge/testdata/solana-whitepaper-en.pdf")
	require.NoError(t, err)

	agentFile, err := os.ReadFile("./examples/solana_expert.agent.yaml")
	require.NoError(t, err)

	var agent entity.Agent
	err = yaml.Unmarshal(agentFile, &agent)
	require.NoError(t, err)

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))

	knowledgeConfig := config.NewKnowledgeConfig()
	knowledgeConfig.SqlitePath = ":memory:"

	knowledgeService, err := knowledge.NewService(ctx, &config.ModelConfig{
		OpenAIAPIKey:    os.Getenv("OPENAI_API_KEY"),
		AnthropicAPIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}, knowledgeConfig, logger)
	require.NoError(t, err)

	if _, err := knowledgeService.IndexKnowledgeFromPDF(ctx, "solana-whitepaper", func(yield func(io.Reader, error) bool) { yield(bytes.NewReader(pdfFile), nil) }); err != nil {
		t.Fatalf("Failed to index knowledge from PDF: %v", err)
	}

	runtime, err := agentruntime.NewAgentRuntime(
		ctx,
		agentruntime.WithAgent(agent),
		agentruntime.WithOpenAIAPIKey(os.Getenv("OPENAI_API_KEY")),
		agentruntime.WithAnthropicAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		agentruntime.WithKnowledgeService(knowledgeService),
		agentruntime.WithLogger(logger),
	)

	require.NoError(t, err)
	defer runtime.Close()

	var out string
	resp, err := runtime.Run(ctx, engine.RunRequest{
		History: []engine.Conversation{
			{
				User: "USER",
				Text: "What is Solana? Can you explain the details to me?",
			},
		},
	}, nil)
	require.NoError(t, err)
	require.NotNil(t, resp)

	out = resp.Text()
	t.Logf("Response: %+v", resp)
	t.Logf("Output: %s", out)

	require.True(t, strings.Contains(out, "Solana"), "Output should contain `Solana`")
}
