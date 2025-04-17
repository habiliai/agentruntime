package jsonrpc

import (
	"fmt"
)

// RuntimeClient provides client methods for runtime service
type RuntimeClient struct {
	client *Client
}

// NewRuntimeClient creates a new runtime client
func NewRuntimeClient(addr string) (*RuntimeClient, error) {
	client, err := NewClient(addr)
	if err != nil {
		return nil, err
	}
	return &RuntimeClient{client: client}, nil
}

// Run executes the specified agents in a thread
func (c *RuntimeClient) Run(threadID uint, agentNames []string) error {
	req := &RunRequest{
		ThreadID:   threadID,
		AgentNames: agentNames,
	}
	resp := &RunResponse{}
	return c.client.Call("RuntimeService.Run", req, resp)
}

// Close closes the client connection
func (c *RuntimeClient) Close() error {
	return c.client.Close()
}

// ThreadClient provides client methods for thread service
type ThreadClient struct {
	client *Client
}

// NewThreadClient creates a new thread client
func NewThreadClient(addr string) (*ThreadClient, error) {
	client, err := NewClient(addr)
	if err != nil {
		return nil, err
	}
	return &ThreadClient{client: client}, nil
}

// CreateThread creates a new thread
func (c *ThreadClient) CreateThread(instruction string, metadata map[string]string) (uint, error) {
	req := &CreateThreadRequest{
		Instruction: instruction,
		Metadata:    metadata,
	}
	resp := &CreateThreadResponse{}
	if err := c.client.Call("ThreadService.CreateThread", req, resp); err != nil {
		return 0, err
	}
	return resp.ThreadID, nil
}

// GetThread retrieves thread information
func (c *ThreadClient) GetThread(threadID uint) (*Thread, error) {
	req := &GetThreadRequest{
		ThreadID: threadID,
	}
	resp := &Thread{}
	if err := c.client.Call("ThreadService.GetThread", req, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// AddMessage adds a message to a thread
func (c *ThreadClient) AddMessage(threadID uint, sender, content string, toolCalls []MessageToolCall) (uint, error) {
	req := &AddMessageRequest{
		ThreadID:  threadID,
		Sender:    sender,
		Content:   content,
		ToolCalls: toolCalls,
	}
	resp := &AddMessageResponse{}
	if err := c.client.Call("ThreadService.AddMessage", req, resp); err != nil {
		return 0, err
	}
	return resp.MessageID, nil
}

// GetMessages retrieves messages from a thread
func (c *ThreadClient) GetMessages(threadID uint, order string, cursor, limit uint) ([]Message, error) {
	req := &GetMessagesRequest{
		ThreadID: threadID,
		Order:    order,
		Cursor:   cursor,
		Limit:    limit,
	}
	resp := &GetMessagesResponse{}
	if err := c.client.Call("ThreadService.GetMessages", req, resp); err != nil {
		return nil, err
	}
	return resp.Messages, nil
}

// GetNumMessages retrieves the number of messages in a thread
func (c *ThreadClient) GetNumMessages(threadID uint) (uint, error) {
	req := &GetNumMessagesRequest{
		ThreadID: threadID,
	}
	resp := &GetNumMessagesResponse{}
	if err := c.client.Call("ThreadService.GetNumMessages", req, resp); err != nil {
		return 0, err
	}
	return resp.NumMessages, nil
}

// Close closes the client connection
func (c *ThreadClient) Close() error {
	return c.client.Close()
}

// NetworkClient provides client methods for network service
type NetworkClient struct {
	client *Client
}

// NewNetworkClient creates a new network client
func NewNetworkClient(addr string) (*NetworkClient, error) {
	client, err := NewClient(addr)
	if err != nil {
		return nil, err
	}
	return &NetworkClient{client: client}, nil
}

// GetAgentRuntimeInfo retrieves agent runtime information
func (c *NetworkClient) GetAgentRuntimeInfo(names []string, all bool) ([]AgentRuntimeInfo, error) {
	req := &GetAgentRuntimeInfoRequest{
		Names: names,
		All:   all,
	}
	resp := &GetAgentRuntimeInfoResponse{}
	if err := c.client.Call("NetworkService.GetAgentRuntimeInfo", req, resp); err != nil {
		return nil, err
	}
	return resp.AgentRuntimeInfo, nil
}

// RegisterAgent registers an agent
func (c *NetworkClient) RegisterAgent(addr string, secure bool, names []string) error {
	req := &RegisterAgentRequest{
		Addr:   addr,
		Secure: secure,
		Names:  names,
	}
	resp := &RegisterAgentResponse{}
	return c.client.Call("NetworkService.RegisterAgent", req, resp)
}

// DeregisterAgent deregisters an agent
func (c *NetworkClient) DeregisterAgent(names []string) error {
	req := &DeregisterAgentRequest{
		Names: names,
	}
	resp := &DeregisterAgentResponse{}
	return c.client.Call("NetworkService.DeregisterAgent", req, resp)
}

// CheckLive checks if agents are live
func (c *NetworkClient) CheckLive(names []string) error {
	req := &CheckLiveRequest{
		Names: names,
	}
	resp := &CheckLiveResponse{}
	return c.client.Call("NetworkService.CheckLive", req, resp)
}

// Close closes the client connection
func (c *NetworkClient) Close() error {
	return c.client.Close()
}