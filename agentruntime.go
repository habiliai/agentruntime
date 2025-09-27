package agentruntime

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/firebase/genkit/go/ai"
	"github.com/habiliai/agentruntime/config"
	"github.com/habiliai/agentruntime/engine"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/genkit"
	"github.com/habiliai/agentruntime/internal/mylog"
	"github.com/habiliai/agentruntime/knowledge"
	"github.com/habiliai/agentruntime/memory"
	"github.com/habiliai/agentruntime/tool"
)

type (
	AgentRuntime struct {
		engine           *engine.Engine
		toolManager      tool.Manager
		logger           *slog.Logger
		agent            *entity.Agent
		knowledgeService knowledge.Service
		memoryService    memory.Service

		modelConfig     *config.ModelConfig
		knowledgeConfig *config.KnowledgeConfig
		logConfig       *config.LogConfig
		memoryConfig    *config.MemoryConfig
	}
	Option func(*AgentRuntime)
)

func (r *AgentRuntime) Agent() *entity.Agent {
	return r.agent
}

func (r *AgentRuntime) Generate(ctx context.Context, req engine.GenerateRequest, opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
	return r.engine.Generate(ctx, &req, opts...)
}

func (r *AgentRuntime) Embed(ctx context.Context, texts ...string) ([][]float32, error) {
	return r.engine.Embed(ctx, texts...)
}

func (r *AgentRuntime) Run(ctx context.Context, req engine.RunRequest, streamCallback ai.ModelStreamCallback) (*engine.RunResponse, error) {
	return r.engine.Run(ctx, *r.agent, req, streamCallback)
}

func (r *AgentRuntime) Close() {
	r.toolManager.Close()
}

func NewAgentRuntime(ctx context.Context, optionFuncs ...Option) (*AgentRuntime, error) {
	e := &AgentRuntime{
		modelConfig:     &config.ModelConfig{},
		knowledgeConfig: config.NewKnowledgeConfig(),
		logConfig:       config.NewLogConfig(),
		memoryConfig:    config.NewMemoryConfig(),
	}
	for _, f := range optionFuncs {
		f(e)
	}

	if e.logger == nil {
		e.logger = mylog.NewLogger(e.logConfig.LogLevel, e.logConfig.LogHandler)
	}

	if e.agent == nil {
		return nil, errors.New("agent is required")
	}

	if e.modelConfig == nil {
		return nil, errors.New("model config is required")
	}

	g := genkit.NewGenkit(ctx, e.modelConfig, e.logger, e.modelConfig.TraceVerbose)

	var err error
	if e.knowledgeService == nil {
		e.knowledgeService, err = knowledge.NewServiceWithStore(ctx, e.knowledgeConfig, e.modelConfig, e.logger, knowledge.NewInMemoryStore())
		if err != nil {
			return nil, err
		}
	}

	if e.memoryService == nil {
		e.memoryService, err = memory.NewService(ctx, e.modelConfig, e.memoryConfig, e.logger)
		if err != nil {
			return nil, err
		}
	}

	e.toolManager, err = tool.NewToolManager(ctx, e.agent.Skills, e.logger, g, e.knowledgeService, e.memoryService)
	if err != nil {
		return nil, err
	}

	if len(e.agent.Knowledge) > 0 {
		// Index knowledge for RAG if available
		knowledgeId := fmt.Sprintf("%s-knowledge", e.agent.Name)
		if _, err := e.knowledgeService.IndexKnowledgeFromMap(ctx, knowledgeId, e.agent.Knowledge); err != nil {
			e.logger.Warn("failed to index knowledge for agent - agent will work without RAG functionality",
				"agent", e.agent.Name,
				"error", err)
			// Continue without failing agent creation
		}
	}

	// Create engine with conversation summarizer if configured
	if e.modelConfig.ConversationSummary.MaxTokens > 0 {
		var err error
		e.engine, err = engine.NewEngineWithSummarizer(
			e.logger,
			e.toolManager,
			g,
			e.modelConfig,
		)
		if err != nil {
			e.logger.Warn("failed to create engine with summarizer, falling back to default", "error", err)
			e.engine = engine.NewEngine(
				e.logger,
				e.toolManager,
				g,
			)
		}
	} else {
		e.engine = engine.NewEngine(
			e.logger,
			e.toolManager,
			g,
		)
	}

	return e, nil
}

func (r *AgentRuntime) GetToolManager() tool.Manager {
	return r.toolManager
}

func (r *AgentRuntime) GetMemoryService() memory.Service {
	return r.memoryService
}

func (r *AgentRuntime) EstimateTokens(ctx context.Context, agent entity.Agent, req engine.RunRequest) (int, error) {
	return r.engine.EstimateTokens(ctx, agent, req)
}

func WithOpenAIAPIKey(apiKey string) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.modelConfig.OpenAIAPIKey = apiKey
	}
}

func WithLogger(logger *slog.Logger) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.logger = logger
	}
}

func WithTraceVerbose(traceVerbose bool) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.modelConfig.TraceVerbose = traceVerbose
	}
}

func WithXAIAPIKey(apiKey string) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.modelConfig.XAIAPIKey = apiKey
	}
}

func WithAnthropicAPIKey(apiKey string) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.modelConfig.AnthropicAPIKey = apiKey
	}
}

func WithLogConfig(logConfig *config.LogConfig) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.logConfig = logConfig
	}
}

func WithAgent(agent entity.Agent) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.agent = &agent
	}
}

func WithKnowledgeService(knowledgeService knowledge.Service) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.knowledgeService = knowledgeService
	}
}

func WithMemoryService(memoryService memory.Service) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.memoryService = memoryService
	}
}

// WithConversationSummary sets the conversation summarization configuration
func WithConversationSummary(summaryConfig config.ConversationSummaryConfig) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.modelConfig.ConversationSummary = summaryConfig
	}
}

// WithDefaultConversationSummary enables conversation summarization with default settings
func WithDefaultConversationSummary() func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		e.modelConfig.ConversationSummary = config.DefaultConversationSummaryConfig()
	}
}

// WithConversationSummaryTokenLimit sets only the token limit for conversation summarization
func WithConversationSummaryTokenLimit(maxTokens int) func(e *AgentRuntime) {
	return func(e *AgentRuntime) {
		if e.modelConfig.ConversationSummary.MaxTokens == 0 {
			// Use default config and override max tokens
			e.modelConfig.ConversationSummary = config.DefaultConversationSummaryConfig()
		}
		e.modelConfig.ConversationSummary.MaxTokens = maxTokens
	}
}
