package genkit

import (
	"log/slog"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/openai"
	"github.com/habiliai/agentruntime/config"
	"github.com/habiliai/agentruntime/errors"
	"github.com/habiliai/agentruntime/internal/genkit/plugins/xai"
	"github.com/habiliai/agentruntime/internal/mylog"
	"github.com/jcooky/go-din"
)

var (
	Key = din.NewRandomName()
)

func init() {
	din.Register(Key, func(c *din.Container) (any, error) {
		var (
			plugins      []genkit.Plugin
			defaultModel string
		)
		{
			conf := din.MustGetT[*config.OpenAIConfig](c)
			if conf.APIKey != "" {
				plugins = append(plugins, &openai.OpenAI{
					APIKey: conf.APIKey,
				})
				defaultModel = "openai/gpt-4o"
			}
		}
		{
			conf := din.MustGetT[*config.XAIConfig](c)
			if conf.APIKey != "" {
				plugins = append(plugins, &xai.Plugin{
					APIKey: conf.APIKey,
				})
				defaultModel = "xai/grok-3"
			}
		}
		logConf := din.MustGetT[*config.LogConfig](c)
		logger := din.MustGet[*slog.Logger](c, mylog.Key)
		g, err := genkit.Init(
			c,
			genkit.WithPlugins(plugins...),
			genkit.WithDefaultModel(defaultModel),
		)

		genkit.RegisterSpanProcessor(g,
			&loggingSpanProcessor{
				verbose: logConf.TraceVerbose,
				logger:  logger,
			},
		)

		return g, errors.Wrapf(err, "failed to init genkit")
	})
}
