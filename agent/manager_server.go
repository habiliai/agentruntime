package agent

import (
	"context"
	"github.com/habiliai/agentruntime/internal/di"
)

type managerServer struct {
	UnsafeAgentManagerServer

	manager Manager
}

func (m *managerServer) GetAgentByName(ctx context.Context, request *GetAgentRequest) (*GetAgentResponse, error) {
	agent, err := m.manager.FindAgentByName(ctx, request.Name)
	if err != nil {
		return nil, err
	}

	return &GetAgentResponse{
		Agent: &Agent{
			Id:        uint32(agent.ID),
			Name:      agent.Name,
			ModelName: agent.ModelName,
			Busy:      agent.Busy,
			Metadata:  agent.Metadata.Data(),
		},
	}, nil
}

func (m *managerServer) UpdateAgent(ctx context.Context, req *UpdateAgentRequest) (*UpdateAgentResponse, error) {
	if err := m.manager.UpdateAgent(ctx, uint(req.AgentId), req.Metadata); err != nil {
		return nil, err
	}

	return &UpdateAgentResponse{}, nil
}

var (
	_                AgentManagerServer = (*managerServer)(nil)
	ManagerServerKey                    = di.NewKey()
)

func init() {
	di.Register(ManagerServerKey, func(c context.Context, _ di.Env) (any, error) {
		return &managerServer{
			manager: di.MustGet[Manager](c, ManagerKey),
		}, nil
	})
}
