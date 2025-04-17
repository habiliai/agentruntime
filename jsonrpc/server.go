package jsonrpc

import (
	"context"
	"fmt"
	"github.com/habiliai/agentruntime/internal/di"
	"github.com/habiliai/agentruntime/internal/mylog"
	"log"
)

// StartServer starts the JSON-RPC server with all services registered
func StartServer(ctx context.Context, addr string) (*Server, error) {
	logger := mylog.FromContext(ctx)

	// Create server
	server := di.MustGet[*Server](ctx, ServerKey)

	// Register services
	runtimeService := di.MustGet[*RuntimeService](ctx, RuntimeServiceKey)
	if err := server.RegisterService(runtimeService); err != nil {
		return nil, fmt.Errorf("failed to register runtime service: %w", err)
	}

	threadService := di.MustGet[*ThreadService](ctx, ThreadServiceKey)
	if err := server.RegisterService(threadService); err != nil {
		return nil, fmt.Errorf("failed to register thread service: %w", err)
	}

	networkService := di.MustGet[*NetworkService](ctx, NetworkServiceKey)
	if err := server.RegisterService(networkService); err != nil {
		return nil, fmt.Errorf("failed to register network service: %w", err)
	}

	// Start the server
	if err := server.Start(addr); err != nil {
		return nil, fmt.Errorf("failed to start JSON-RPC server: %w", err)
	}

	logger.Info("JSON-RPC server started", "addr", addr)

	// Handle server shutdown
	go func() {
		<-ctx.Done()
		logger.Info("Shutting down JSON-RPC server")
		if err := server.Stop(); err != nil {
			logger.Error("Failed to stop JSON-RPC server", "error", err)
		}
	}()

	return server, nil
}

// CreateClient creates a JSON-RPC client for the given address
func CreateClient(addr string) (*Client, error) {
	return NewClient(addr)
}