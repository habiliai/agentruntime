package anthropic

import (
	"encoding/json"

	"github.com/firebase/genkit/go/ai"
)

type AnthropicConfig struct {
	ai.GenerationCommonConfig
	ExtendedThinkingEnabled     bool    `json:"extendedThinkingEnabled,omitempty"`
	ExtendedThinkingBudgetRatio float64 `json:"extendedThinkingBudgetRatio,omitempty"`

	WebSearchConfig           *WebSearchConfig `json:"webSearch,omitempty"`
	EnableContext1M           bool             `json:"enableContext1M,omitempty"`
	EnableInterleavedThinking bool             `json:"enableInterleavedThinking,omitempty"`
}

type WebSearchConfig struct {
	MaxUses int64 `json:"maxUses,omitempty"`
}

func (c *AnthropicConfig) Unmarshal(data any) error {
	if data == nil {
		return nil
	}

	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(jsonBytes, c); err != nil {
		return err
	}

	if c.WebSearchConfig != nil {
		if c.WebSearchConfig.MaxUses == 0 {
			c.WebSearchConfig.MaxUses = 99
		}
	}

	return nil
}
