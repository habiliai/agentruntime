package xai

import (
	"context"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go/option"
)

const (
	provider  = "xai"
	apiKeyEnv = "XAI_API_KEY"
	baseUrl   = "https://api.x.ai/v1"
)

var (
	grok3 = ai.ModelOptions{
		Label:    "XAI - grok-3",
		Supports: &compat_oai.BasicText,
		Stage:    ai.ModelStageStable,
		Versions: []string{"grok-3"},
	}
	grok3Mini = ai.ModelOptions{
		Label:    "XAI - grok-3-mini",
		Supports: &compat_oai.BasicText,
		Stage:    ai.ModelStageStable,
		Versions: []string{"grok-3-mini"},
	}
	grok4 = ai.ModelOptions{
		Label:    "XAI - grok-4",
		Supports: &compat_oai.Multimodal,
		Stage:    ai.ModelStageStable,
		Versions: []string{"grok-4", "grok-4-latest"},
	}
	grok4FastReasoning = ai.ModelOptions{
		Label:    "XAI - grok-4-fast-reasoning",
		Supports: &compat_oai.Multimodal,
		Stage:    ai.ModelStageStable,
		Versions: []string{"grok-4-fast-reasoning", "grok-4-fast-reasoning-latest"},
	}
	grok4FastNonReasoning = ai.ModelOptions{
		Label:    "XAI - grok-4-fast-non-reasoning",
		Supports: &compat_oai.Multimodal,
		Stage:    ai.ModelStageStable,
		Versions: []string{"grok-4-fast-non-reasoning", "grok-4-fast-non-reasoning-latest"},
	}
)

type XAI struct {
	// The API key to access the service for XAI.
	// If empty, the values of the environment variables XAI_API_KEY will be consulted.
	APIKey string

	oai *compat_oai.OpenAICompatible
}

var (
	_ api.Plugin = (*XAI)(nil)
)

// Name implements genkit.Plugin.
func (x *XAI) Name() string {
	return provider
}

// Init implements genkit.Plugin.
// After calling Init, you may call [DefineModel] to create and register any additional generative models.
func (x *XAI) Init(ctx context.Context) []api.Action {
	apiKey := x.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(apiKeyEnv)
		if apiKey == "" {
			panic("XAI API key not found in environment variable")
		}
	}

	x.oai = &compat_oai.OpenAICompatible{
		Opts: []option.RequestOption{
			option.WithBaseURL(baseUrl),
			option.WithAPIKey(apiKey),
		},
	}
	actions := x.oai.Init(ctx)

	return append(actions,
		x.oai.DefineModel(provider, "grok-3", grok3).(api.Action),
		x.oai.DefineModel(provider, "grok-3-mini", grok3Mini).(api.Action),
		x.oai.DefineModel(provider, "grok-4", grok4).(api.Action),
		x.oai.DefineModel(provider, "grok-4-fast-reasoning", grok4FastReasoning).(api.Action),
		x.oai.DefineModel(provider, "grok-4-fast-non-reasoning", grok4FastNonReasoning).(api.Action),
	)
}

// Model returns the [ai.Model] with the given name.
// It returns nil if the model was not defined.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}
