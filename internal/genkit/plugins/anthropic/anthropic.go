package anthropic

import (
	"context"
	"os"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const (
	provider          = "anthropic"
	labelPrefix       = "Anthropic"
	apiKeyEnv         = "ANTHROPIC_API_KEY"
	defaultMaxRetries = 4
)

var (
	basicCap = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      true,
	}

	knownCaps = map[string]ai.ModelSupports{
		"claude-sonnet-4-5-20250929": basicCap,
		"claude-opus-4-20250514":     basicCap,
		"claude-sonnet-4-20250514":   basicCap,
		"claude-3-7-sonnet-latest":   basicCap,
		"claude-3-5-haiku-latest":    basicCap,
	}
	defaultRequestTimeout = 10 * time.Minute
	defaultModelParams    = map[string]AnthropicConfig{
		"claude-sonnet-4-5-20250929": {
			GenerationCommonConfig: ai.GenerationCommonConfig{
				MaxOutputTokens: 64_000,
			},
			ExtendedThinkingEnabled:     false,
			ExtendedThinkingBudgetRatio: 0, // Will be calculated dynamically based on actual maxTokens
		},
		"claude-opus-4-20250514": {
			GenerationCommonConfig: ai.GenerationCommonConfig{
				MaxOutputTokens: 32_000,
			},
			ExtendedThinkingEnabled:     true,
			ExtendedThinkingBudgetRatio: 0.15,
		},
		"claude-sonnet-4-20250514": {
			GenerationCommonConfig: ai.GenerationCommonConfig{
				MaxOutputTokens: 64_000,
			},
			ExtendedThinkingEnabled:     true,
			ExtendedThinkingBudgetRatio: 0.15,
		},
		"claude-3-7-sonnet-latest": {
			GenerationCommonConfig: ai.GenerationCommonConfig{
				MaxOutputTokens: 64_000,
			},
			ExtendedThinkingEnabled:     true,
			ExtendedThinkingBudgetRatio: 0.15,
		},
		"claude-3-5-haiku-latest": {
			GenerationCommonConfig: ai.GenerationCommonConfig{
				MaxOutputTokens: 8192,
			},
			ExtendedThinkingEnabled:     false,
			ExtendedThinkingBudgetRatio: 0, // Will be calculated dynamically based on actual maxTokens
		},
	}
)

type Anthropic struct {
	// The API key to access the service for Anthropic.
	// If empty, the values of the environment variables ANTHROPIC_API_KEY will be consulted.
	APIKey string

	// The timeout for requests to the Anthropic API.
	// If empty, the default timeout of 10 minutes will be used.
	RequestTimeout time.Duration

	// The maximum number of retries for the request.
	// If empty, the default value of 3 will be used.
	MaxRetries int

	// The client to use for the Anthropic API.
	client anthropic.Client
}

var (
	_ api.Plugin = (*Anthropic)(nil)
)

// Name implements genkit.Plugin.
func (a *Anthropic) Name() string {
	return provider
}

// Init implements genkit.Plugin.
// After calling Init, you may call [DefineModel] to create and register any additional generative models.
func (a *Anthropic) Init(_ context.Context) (actions []api.Action) {
	apiKey := a.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(apiKeyEnv)
		if apiKey == "" {
			panic("the Anthropic API key not found in environment variable")
		}
	}

	if a.RequestTimeout == 0 {
		a.RequestTimeout = defaultRequestTimeout
	}
	if a.MaxRetries == 0 {
		a.MaxRetries = defaultMaxRetries
	}

	a.client = anthropic.NewClient(
		option.WithAPIKey(apiKey),
		option.WithRequestTimeout(a.RequestTimeout),
		option.WithMaxRetries(a.MaxRetries),
		option.WithEnvironmentProduction(),
	)

	// Define models with simplified names as requested
	actions = append(actions,
		a.DefineModel(labelPrefix, provider, "claude-4-opus", "claude-opus-4-20250514", knownCaps["claude-opus-4-20250514"]).(api.Action),
		a.DefineModel(labelPrefix, provider, "claude-4-sonnet", "claude-sonnet-4-20250514", knownCaps["claude-sonnet-4-20250514"]).(api.Action),
		// Also define Claude 3.7 and 3.5 models as alternatives
		a.DefineModel(labelPrefix, provider, "claude-3.7-sonnet", "claude-3-7-sonnet-latest", knownCaps["claude-3-7-sonnet-latest"]).(api.Action),
		a.DefineModel(labelPrefix, provider, "claude-3.5-haiku", "claude-3-5-haiku-latest", knownCaps["claude-3-5-haiku-latest"]).(api.Action),
		a.DefineModel(labelPrefix, provider, "claude-4.5-sonnet", "claude-sonnet-4-5-20250929", knownCaps["claude-sonnet-4-5-20250929"]).(api.Action),
	)

	return
}

// Model returns the [ai.Model] with the given name.
// It returns nil if the model was not defined.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, provider+"/"+name)
}
