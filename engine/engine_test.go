package engine_test

import (
	_ "embed"
	"log/slog"
	"os"
	"testing"

	"github.com/habiliai/agentruntime/config"
	"github.com/habiliai/agentruntime/engine"
	genkitinternal "github.com/habiliai/agentruntime/internal/genkit"
	"github.com/habiliai/agentruntime/internal/mytesting"
	"github.com/stretchr/testify/suite"
)

type EngineTestSuite struct {
	mytesting.Suite

	engine *engine.Engine
}

func (s *EngineTestSuite) SetupTest() {
	s.Suite.SetupTest()

	g := genkitinternal.NewGenkit(s, &config.ModelConfig{
		OpenAIAPIKey:    os.Getenv("OPENAI_API_KEY"),
		XAIAPIKey:       os.Getenv("XAI_API_KEY"),
		AnthropicAPIKey: os.Getenv("ANTHROPIC_API_KEY"),
	}, slog.Default(), true)

	s.engine = engine.NewEngine(
		slog.Default(),
		nil,
		g,
	)
}

func (s *EngineTestSuite) TearDownTest() {
	s.Suite.TearDownTest()
}

func TestRunner(t *testing.T) {
	suite.Run(t, new(EngineTestSuite))
}
