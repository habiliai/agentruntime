package config

import (
	"github.com/jcooky/go-din"
)

type RuntimeConfig struct {
	LogConfig
	OpenAIConfig
	ToolConfig
	Host           string `env:"HOST"`
	Port           int    `env:"PORT"`
	NetworkBaseUrl string `env:"NETWORK_BASE_URL"`
	RuntimeBaseUrl string `env:"RUNTIME_BASE_URL"`
}

func init() {
	din.RegisterT(func(c *din.Container) (*RuntimeConfig, error) {
		conf := &RuntimeConfig{
			LogConfig: LogConfig{
				LogLevel:   "debug",
				LogHandler: "default",
			},
			Host:           "0.0.0.0",
			Port:           10080,
			NetworkBaseUrl: "http://127.0.0.1:9080/rpc",
			RuntimeBaseUrl: "http://127.0.0.1:10080/rpc",
		}

		if err := resolveConfig(conf, c.Env == din.EnvTest); err != nil {
			return nil, err
		}

		return conf, nil
	})
}
