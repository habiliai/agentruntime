package jsonrpc

import (
	"context"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/di"
	"github.com/habiliai/agentruntime/network"
)

// NetworkService provides JSON-RPC compatible methods for network operations
type NetworkService struct {
	service network.Service
}

// GetAgentRuntimeInfoRequest represents a request to get agent runtime info
type GetAgentRuntimeInfoRequest struct {
	Names []string `json:"names"`
	All   bool     `json:"all,omitempty"`
}

// AgentRuntimeInfo represents information about an agent runtime
type AgentRuntimeInfo struct {
	Addr       string   `json:"addr"`
	Secure     bool     `json:"secure"`
	AgentNames []string `json:"agent_names"`
}

// GetAgentRuntimeInfoResponse represents the response for agent runtime info
type GetAgentRuntimeInfoResponse struct {
	AgentRuntimeInfo []AgentRuntimeInfo `json:"agent_runtime_info"`
}

// RegisterAgentRequest represents a request to register an agent
type RegisterAgentRequest struct {
	Addr  string   `json:"addr"`
	Secure bool     `json:"secure"`
	Names []string `json:"names"`
}

// RegisterAgentResponse is empty for consistency
type RegisterAgentResponse struct{}

// DeregisterAgentRequest represents a request to deregister an agent
type DeregisterAgentRequest struct {
	Names []string `json:"names"`
}

// DeregisterAgentResponse is empty for consistency
type DeregisterAgentResponse struct{}

// CheckLiveRequest represents a request to check if agents are live
type CheckLiveRequest struct {
	Names []string `json:"names"`
}

// CheckLiveResponse is empty for consistency
type CheckLiveResponse struct{}

// GetAgentRuntimeInfo retrieves agent runtime information
func (s *NetworkService) GetAgentRuntimeInfo(req *GetAgentRuntimeInfoRequest, resp *GetAgentRuntimeInfoResponse) error {
	ctx := context.Background()
	
	var (
		runtimeInfo []entity.AgentRuntime
		err         error
	)
	
	if req.All {
		runtimeInfo, err = s.service.GetAllAgentRuntimeInfo(ctx)
	} else {
		runtimeInfo, err = s.service.GetAgentRuntimeInfo(ctx, req.Names)
	}
	
	if err != nil {
		return err
	}

	// Process agent runtime info
	infoMap := make(map[string]*AgentRuntimeInfo)
	for _, info := range runtimeInfo {
		key := info.Addr
		if rtInfo, ok := infoMap[key]; ok {
			rtInfo.AgentNames = append(rtInfo.AgentNames, info.Name)
		} else {
			infoMap[key] = &AgentRuntimeInfo{
				Addr:       info.Addr,
				Secure:     info.Secure,
				AgentNames: []string{info.Name},
			}
		}
	}

	// Convert map to slice
	for _, info := range infoMap {
		resp.AgentRuntimeInfo = append(resp.AgentRuntimeInfo, *info)
	}

	return nil
}

// RegisterAgent registers an agent
func (s *NetworkService) RegisterAgent(req *RegisterAgentRequest, resp *RegisterAgentResponse) error {
	ctx := context.Background()
	return s.service.RegisterAgent(ctx, req.Addr, req.Secure, req.Names)
}

// DeregisterAgent deregisters an agent
func (s *NetworkService) DeregisterAgent(req *DeregisterAgentRequest, resp *DeregisterAgentResponse) error {
	ctx := context.Background()
	return s.service.DeregisterAgent(ctx, req.Names)
}

// CheckLive checks if agents are live
func (s *NetworkService) CheckLive(req *CheckLiveRequest, resp *CheckLiveResponse) error {
	ctx := context.Background()
	return s.service.CheckLive(ctx, req.Names)
}

var (
	NetworkServiceKey = di.NewKey()
)

func init() {
	di.Register(NetworkServiceKey, func(ctx context.Context, _ di.Env) (any, error) {
		return &NetworkService{
			service: di.MustGet[network.Service](ctx, network.ManagerKey),
		}, nil
	})
}