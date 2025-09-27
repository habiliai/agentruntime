package tool_test

import (
	"log/slog"
	"os"
	"testing"

	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/genkit"
	"github.com/habiliai/agentruntime/internal/mytesting"
	"github.com/habiliai/agentruntime/tool"
	"github.com/stretchr/testify/suite"
)

type TestSuite struct {
	mytesting.Suite

	toolManager tool.Manager
}

func (s *TestSuite) SetupTest() {
	s.Suite.SetupTest()

	g := genkit.NewGenkit(
		s,
		nil,
		slog.Default(),
		false,
	)
	var err error
	s.toolManager, err = tool.NewToolManager(
		s,
		[]entity.AgentSkillUnion{
			{
				Type: "mcp",
				OfMCP: &entity.MCPAgentSkill{
					Name:    "filesystem",
					Command: "npx",
					Args: []string{
						"-y", "@modelcontextprotocol/server-filesystem", ".",
					},
				},
			},
			{
				Type: "nativeTool",
				OfNative: &entity.NativeAgentSkill{
					Name:    "get_weather",
					Details: "Get weather information when you need it",
					Env: map[string]any{
						"OPENWEATHER_API_KEY": os.Getenv("OPENWEATHER_API_KEY"),
					},
				},
			},
		},
		slog.Default(),
		g,
		nil,
		nil,
	)
	s.Require().NoError(err)

}

func (s *TestSuite) TearDownTest() {
	s.toolManager.Close()
	s.Suite.TearDownTest()
}

func TestTool(t *testing.T) {
	suite.Run(t, new(TestSuite))
}
