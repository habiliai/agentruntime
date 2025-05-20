package runtime

import (
	"net/http"

	"github.com/gorilla/rpc/v2"
	"github.com/jcooky/go-din"
)

type (
	JsonRpcService struct {
		runtime Service
	}

	RunRequest struct {
		ThreadId   uint32   `json:"thread_id"`
		AgentNames []string `json:"agent_names"`
	}

	RunResponse struct{}
)

func (s *JsonRpcService) Run(r *http.Request, args *RunRequest, _ *RunResponse) error {
	agents, err := s.runtime.FindAgentsByNames(args.AgentNames)
	if err != nil {
		return err
	}
	if err = s.runtime.Run(r.Context(), uint(args.ThreadId), agents); err != nil {
		return err
	}
	return nil
}

var (
	servicePrefix = "habiliai-agentnetwork-v1"
)

func RegisterJsonRpcService(c *din.Container, server *rpc.Server) error {
	svc := &JsonRpcService{
		runtime: din.MustGetT[Service](c),
	}

	return server.RegisterService(svc, servicePrefix)
}
