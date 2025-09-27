package genkit

import (
	"context"
	"log/slog"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/openai"
	"github.com/habiliai/agentruntime/config"
	"github.com/habiliai/agentruntime/internal/genkit/plugins/anthropic"
	"github.com/habiliai/agentruntime/internal/genkit/plugins/xai"
	"github.com/jcooky/go-din"
)

var (
	Key = din.NewRandomName()
)

func NewGenkit(
	ctx context.Context,
	modelConfig *config.ModelConfig,
	logger *slog.Logger,
	traceVerbose bool,
) *genkit.Genkit {
	var (
		plugins      []api.Plugin
		defaultModel string
	)
	{
		if modelConfig != nil && modelConfig.OpenAIAPIKey != "" {
			plugins = append(plugins, &openai.OpenAI{
				APIKey: modelConfig.OpenAIAPIKey,
			})
			defaultModel = "openai/gpt-4o"
			logger.Info("Loaded OpenAI plugin", "model", defaultModel)
		}
	}
	{
		if modelConfig != nil && modelConfig.XAIAPIKey != "" {
			plugins = append(plugins, &xai.XAI{
				APIKey: modelConfig.XAIAPIKey,
			})
			defaultModel = "xai/grok-3"
			logger.Info("Loaded XAI plugin", "model", defaultModel)
		}
	}
	{
		if modelConfig != nil && modelConfig.AnthropicAPIKey != "" {
			plugins = append(plugins, &anthropic.Anthropic{
				APIKey: modelConfig.AnthropicAPIKey,
			})
			defaultModel = "anthropic/claude-4-sonnet"
			logger.Info("Loaded Anthropic plugin", "model", defaultModel)
		}
	}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugins...),
		genkit.WithDefaultModel(defaultModel),
	)

	return g
}
