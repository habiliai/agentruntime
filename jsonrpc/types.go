package jsonrpc

import (
	"context"
	"github.com/habiliai/agentruntime/internal/di"
	"net"
	"net/rpc"
	"net/rpc/jsonrpc"
)

// Server represents a JSON-RPC server
type Server struct {
	rpcServer *rpc.Server
	listener  net.Listener
}

// NewServer creates a new JSON-RPC server
func NewServer() *Server {
	return &Server{
		rpcServer: rpc.NewServer(),
	}
}

// RegisterService registers a service with the RPC server
func (s *Server) RegisterService(service interface{}) error {
	return s.rpcServer.Register(service)
}

// Start starts the JSON-RPC server on the given address
func (s *Server) Start(addr string) error {
	var err error
	s.listener, err = net.Listen("tcp", addr)
	if err != nil {
		return err
	}

	go func() {
		for {
			conn, err := s.listener.Accept()
			if err != nil {
				// Server closed
				return
			}
			go s.rpcServer.ServeCodec(jsonrpc.NewServerCodec(conn))
		}
	}()

	return nil
}

// Stop stops the JSON-RPC server
func (s *Server) Stop() error {
	if s.listener != nil {
		return s.listener.Close()
	}
	return nil
}

// Client represents a JSON-RPC client
type Client struct {
	client *rpc.Client
}

// NewClient creates a new JSON-RPC client
func NewClient(addr string) (*Client, error) {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return nil, err
	}

	return &Client{
		client: jsonrpc.NewClient(conn),
	}, nil
}

// Call invokes a remote method
func (c *Client) Call(serviceMethod string, args interface{}, reply interface{}) error {
	return c.client.Call(serviceMethod, args, reply)
}

// Close closes the client connection
func (c *Client) Close() error {
	return c.client.Close()
}

var (
	ServerKey = di.NewKey()
)

func init() {
	di.Register(ServerKey, func(ctx context.Context, _ di.Env) (any, error) {
		server := NewServer()
		return server, nil
	})
}