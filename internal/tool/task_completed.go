package tool

import (
	"context"
)

type (
	DoneAgentRequest struct {
		Reason string `json:"reason" jsonschema:"description=Reason why the task is considered done"`
	}

	DoneAgentResponse struct {
		Success bool   `json:"success"`
		Message string `json:"message"`
	}
)

func (m *manager) DoneAgent(_ context.Context, req *DoneAgentRequest) (*DoneAgentResponse, error) {
	// Verify thread exists
	return &DoneAgentResponse{
		Success: true,
		Message: "Task marked as completed: " + req.Reason,
	}, nil
}

func (m *manager) registerDoneTool() {
	registerLocalTool(
		m,
		"done_agent",
		"Mark the current task as completed when you've fulfilled all requirements",
		func(ctx context.Context, req struct {
			*DoneAgentRequest
		}) (res struct {
			*DoneAgentResponse
		}, err error) {
			res.DoneAgentResponse, err = m.DoneAgent(ctx, req.DoneAgentRequest)
			return
		},
	)
}
