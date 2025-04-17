package jsonrpc

import (
	"context"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/di"
	"github.com/habiliai/agentruntime/runtime"
)

// RuntimeService provides JSON-RPC compatible methods for the runtime service
type RuntimeService struct {
	service runtime.Service
}

// RunRequest represents the request for Run
type RunRequest struct {
	ThreadID   uint     `json:"thread_id"`
	AgentNames []string `json:"agent_names"`
}

// RunResponse represents the response for Run
type RunResponse struct {
	// Empty for now, matching gRPC structure
}

// Run starts agent execution
func (s *RuntimeService) Run(req *RunRequest, resp *RunResponse) error {
	agents, err := s.service.findAgentsByNames(req.AgentNames)
	if err != nil {
		return err
	}
	
	// Use background context since RPC doesn't pass context
	ctx := context.Background()
	return s.service.Run(ctx, req.ThreadID, agents)
}

var (
	RuntimeServiceKey = di.NewKey()
)

func init() {
	di.Register(RuntimeServiceKey, func(ctx context.Context, _ di.Env) (any, error) {
		return &RuntimeService{
			service: di.MustGet[runtime.Service](ctx, runtime.ServiceKey),
		}, nil
	})
}