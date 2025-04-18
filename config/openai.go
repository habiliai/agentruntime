package config

import (
	"context"
	"github.com/habiliai/agentruntime/internal/di"
)

type OpenAIConfig struct {
	OpenAIApiKey string `env:"OPENAI_API_KEY"`
}

var OpenAIConfigKey = di.NewKey()

func init() {
	di.Register(OpenAIConfigKey, func(ctx context.Context, env di.Env) (any, error) {
		conf := OpenAIConfig{}
		return &conf, resolveConfig(&conf, env == di.EnvTest)
	})
}
