package jsonrpc

import (
	"context"
	"encoding/json"
	"github.com/habiliai/agentruntime/entity"
	"github.com/habiliai/agentruntime/internal/di"
	"github.com/habiliai/agentruntime/thread"
	"github.com/pkg/errors"
	"time"
)

// ThreadService provides JSON-RPC compatible methods for thread operations
type ThreadService struct {
	manager thread.Manager
}

// CreateThreadRequest represents a request to create a thread
type CreateThreadRequest struct {
	Instruction string            `json:"instruction"`
	Metadata    map[string]string `json:"metadata"`
}

// CreateThreadResponse represents the response from creating a thread
type CreateThreadResponse struct {
	ThreadID uint `json:"thread_id"`
}

// GetThreadRequest represents a request to get thread info
type GetThreadRequest struct {
	ThreadID uint `json:"thread_id"`
}

// Thread represents thread information
type Thread struct {
	ID          uint      `json:"id"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	Instruction string    `json:"instruction"`
	Participants []string `json:"participants"`
}

// AddMessageRequest represents a request to add a message
type AddMessageRequest struct {
	ThreadID  uint                `json:"thread_id"`
	Sender    string              `json:"sender"`
	Content   string              `json:"content"`
	ToolCalls []MessageToolCall   `json:"tool_calls,omitempty"`
}

// MessageToolCall represents a tool call in a message
type MessageToolCall struct {
	Name      string      `json:"name"`
	Arguments interface{} `json:"arguments"`
	Result    interface{} `json:"result"`
}

// AddMessageResponse represents the response from adding a message
type AddMessageResponse struct {
	MessageID uint `json:"message_id"`
}

// GetMessagesRequest represents a request to get messages
type GetMessagesRequest struct {
	ThreadID uint   `json:"thread_id"`
	Order    string `json:"order"` // "ASC" or "DESC"
	Cursor   uint   `json:"cursor,omitempty"`
	Limit    uint   `json:"limit"`
}

// Message represents a thread message
type Message struct {
	ID        uint            `json:"id"`
	Content   string          `json:"content"`
	CreatedAt time.Time       `json:"created_at"`
	UpdatedAt time.Time       `json:"updated_at"`
	Sender    string          `json:"sender"`
	ToolCalls []MessageToolCall `json:"tool_calls,omitempty"`
}

// GetMessagesResponse represents the response from getting messages
type GetMessagesResponse struct {
	Messages []Message `json:"messages"`
}

// GetNumMessagesRequest represents a request to get number of messages
type GetNumMessagesRequest struct {
	ThreadID uint `json:"thread_id"`
}

// GetNumMessagesResponse represents the response from getting number of messages
type GetNumMessagesResponse struct {
	NumMessages uint `json:"num_messages"`
}

// CreateThread creates a new thread
func (s *ThreadService) CreateThread(req *CreateThreadRequest, resp *CreateThreadResponse) error {
	ctx := context.Background()
	thr, err := s.manager.CreateThread(ctx, req.Instruction)
	if err != nil {
		return err
	}

	resp.ThreadID = thr.ID
	return nil
}

// GetThread retrieves thread information
func (s *ThreadService) GetThread(req *GetThreadRequest, resp *Thread) error {
	ctx := context.Background()
	thr, err := s.manager.GetThreadById(ctx, req.ThreadID)
	if err != nil {
		return err
	}

	resp.ID = thr.ID
	resp.CreatedAt = thr.CreatedAt
	resp.UpdatedAt = thr.UpdatedAt
	resp.Instruction = thr.Instruction
	resp.Participants = thr.Participants

	return nil
}

// AddMessage adds a message to a thread
func (s *ThreadService) AddMessage(req *AddMessageRequest, resp *AddMessageResponse) error {
	ctx := context.Background()
	
	content := entity.MessageContent{
		Text: req.Content,
	}

	for _, toolCall := range req.ToolCalls {
		content.ToolCalls = append(content.ToolCalls, entity.MessageContentToolCall{
			Name:      toolCall.Name,
			Arguments: toolCall.Arguments,
			Result:    toolCall.Result,
		})
	}

	msg, err := s.manager.AddMessage(ctx, req.ThreadID, req.Sender, content)
	if err != nil {
		return err
	}

	resp.MessageID = msg.ID
	return nil
}

// GetMessages retrieves messages from a thread
func (s *ThreadService) GetMessages(req *GetMessagesRequest, resp *GetMessagesResponse) error {
	ctx := context.Background()
	order := "ASC"
	if req.Order == "DESC" {
		order = "DESC"
	}

	messages, err := s.manager.GetMessages(ctx, req.ThreadID, order, req.Cursor, req.Limit)
	if err != nil {
		return err
	}

	for _, msg := range messages {
		content := msg.Content.Data()
		message := Message{
			ID:        msg.ID,
			Content:   content.Text,
			CreatedAt: msg.CreatedAt,
			UpdatedAt: msg.UpdatedAt,
			Sender:    msg.User,
		}

		for _, toolCall := range content.ToolCalls {
			message.ToolCalls = append(message.ToolCalls, MessageToolCall{
				Name:      toolCall.Name,
				Arguments: toolCall.Arguments,
				Result:    toolCall.Result,
			})
		}

		resp.Messages = append(resp.Messages, message)
	}

	return nil
}

// GetNumMessages retrieves the number of messages in a thread
func (s *ThreadService) GetNumMessages(req *GetNumMessagesRequest, resp *GetNumMessagesResponse) error {
	ctx := context.Background()
	numMessages, err := s.manager.GetNumMessages(ctx, req.ThreadID)
	if err != nil {
		return err
	}

	resp.NumMessages = numMessages
	return nil
}

var (
	ThreadServiceKey = di.NewKey()
)

func init() {
	di.Register(ThreadServiceKey, func(ctx context.Context, _ di.Env) (any, error) {
		return &ThreadService{
			manager: di.MustGet[thread.Manager](ctx, thread.ManagerKey),
		}, nil
	})
}