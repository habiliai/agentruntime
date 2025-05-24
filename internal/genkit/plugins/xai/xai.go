package xai

import (
	"context"
	"fmt"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/internal/genkit/plugins/internal/config"
	"github.com/habiliai/agentruntime/internal/genkit/plugins/internal/openaiapi"
	goopenai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"os"
)

const (
	provider    = "xai"
	labelPrefix = "XAI"
	apiKeyEnv   = "XAI_API_KEY"
	baseUrl     = "https://api.x.ai/v1"
)

var (
	knownCaps = map[string]ai.ModelSupports{
		"grok-3":           config.Multimodal,
		"grok-3-fast":      config.Multimodal,
		"grok-3-mini":      config.BasicText,
		"grok-3-mini-fast": config.BasicText,
		"grok-2-vision":    config.Multimodal,
		"grok-2-image":     config.Multimodal,
	}
)

type Plugin struct {
	// The API key to access the service for XAI.
	// If empty, the values of the environment variables XAI_API_KEY will be consulted.
	APIKey string
}

var (
	_ genkit.Plugin = (*Plugin)(nil)
)

// Name implements genkit.Plugin.
func (o *Plugin) Name() string {
	return provider
}

// Init implements genkit.Plugin.
// After calling Init, you may call [DefineModel] to create and register any additional generative models.
func (o *Plugin) Init(_ context.Context, g *genkit.Genkit) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("%s.Init: %w", provider, err)
		}
	}()

	apiKey := o.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(apiKeyEnv)
		if apiKey == "" {
			return fmt.Errorf("XAI API key not found in environment variable: %s", apiKeyEnv)
		}
	}

	client := goopenai.NewClient(
		option.WithBaseURL(baseUrl),
		option.WithAPIKey(apiKey),
	)

	for model, caps := range knownCaps {
		openaiapi.DefineModel(g, client, labelPrefix, provider, model, caps)
	}

	return nil
}

// Model returns the [ai.Model] with the given name.
// It returns nil if the model was not defined.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, provider, name)
}
