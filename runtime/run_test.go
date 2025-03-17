package runtime_test

import (
	"github.com/habiliai/agentruntime/entity"
	"github.com/mokiat/gog"
)

func (s *AgentRuntimeTestSuite) TestRun() {
	var agents []entity.Agent
	for _, agentConfig := range s.agents {
		agent, err := s.agentManager.SaveAgentFromConfig(s, agentConfig)
		s.Require().NoError(err)

		agents = append(agents, agent)
	}

	thread, err := s.threadManager.CreateThread(s, "# Mission: AI agents dialogue with user")
	s.Require().NoError(err)

	err = s.runtime.Run(s, thread.ID, gog.Map(agents, func(a entity.Agent) uint {
		return a.ID
	}))
	s.Require().NoError(err)

	messages, err := s.threadManager.GetMessages(s, thread.ID, "DESC", 0, 100)
	s.Require().NoError(err)
	s.T().Logf(">> response: %v\n", messages[0].Content.Data().Text)

	s.Require().Equal("done_agent", messages[0].Content.Data().ToolCall.Name)
}
