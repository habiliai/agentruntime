package memory

import (
	"context"
	"log/slog"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/config"
	internalgenkit "github.com/habiliai/agentruntime/internal/genkit"
	"github.com/pkg/errors"
)

type (
	RememberInput struct {
		Key    string   `json:"key"`
		Value  string   `json:"value"`
		Source string   `json:"source"`
		Tags   []string `json:"tags,omitempty"`
	}

	UpdateMemoryInput struct {
		Value *string  `json:"value"`
		Tags  []string `json:"tags,omitempty"`
	}

	Service interface {
		RememberMemory(ctx context.Context, input RememberInput) (*Memory, error)
		SearchMemory(ctx context.Context, query string, limit int) ([]ScoredMemory, error)
		UpdateMemory(ctx context.Context, key string, input UpdateMemoryInput) (*Memory, error)
		GetMemory(ctx context.Context, key string) (*Memory, error)
		DeleteMemory(ctx context.Context, key string) error
		ListMemories(ctx context.Context) ([]*Memory, error)
		GenerateKey(ctx context.Context, input string, tags []string, prompt string, existingKeys []string) (string, error)
		GenerateTags(ctx context.Context, input string, prompt string, existingTags []string) ([]string, error)
	}

	service struct {
		store        Store
		embedder     ai.Embedder
		genkit       *genkit.Genkit
		memoryConfig *config.MemoryConfig
		logger       *slog.Logger
	}
)

var (
	_ Service = (*service)(nil)
)

func NewServiceWithStore(ctx context.Context, store Store, modelConfig *config.ModelConfig, memoryConfig *config.MemoryConfig, logger *slog.Logger) (Service, error) {
	g := internalgenkit.NewGenkit(ctx, modelConfig, logger, modelConfig.TraceVerbose)

	embedder := genkit.LookupEmbedder(g, "openai/text-embedding-3-small")

	return &service{store: store, embedder: embedder, genkit: g, memoryConfig: memoryConfig, logger: logger}, nil
}

func NewService(ctx context.Context, modelConfig *config.ModelConfig, memoryConfig *config.MemoryConfig, logger *slog.Logger) (Service, error) {
	return NewServiceWithStore(ctx, NewInMemoryStore(), modelConfig, memoryConfig, logger)
}

// RememberMemories creates and stores memories from the given inputs
func (s *service) RememberMemory(ctx context.Context, input RememberInput) (*Memory, error) {
	// Generate embedding for the input
	embedding, err := s.embedder.Embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{{Content: []*ai.Part{ai.NewTextPart(input.Value)}}},
	})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to generate embedding for input '%s'", input.Value)
	}

	// Determine source
	source := input.Source
	if source == "" {
		source = MemorySourceUser
	}

	// Create memory with ULID
	memory := &Memory{
		Key:       input.Key,
		Value:     input.Value,
		Source:    MemorySource(source),
		Tags:      input.Tags,
		Embedding: embedding.Embeddings[0].Embedding,
	}

	// Store memory
	if err := s.store.Set(ctx, memory); err != nil {
		return nil, errors.Wrapf(err, "failed to store memory")
	}

	return memory, nil
}

// SearchMemory searches for memories similar to the given query
func (s *service) SearchMemory(ctx context.Context, query string, limit int) ([]ScoredMemory, error) {
	if query == "" {
		return nil, errors.Errorf("query cannot be empty")
	}

	// Generate embedding for the query
	embedding, err := s.embedder.Embed(ctx, &ai.EmbedRequest{
		Input: []*ai.Document{{Content: []*ai.Part{ai.NewTextPart(query)}}},
	})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to generate embedding for query")
	}

	queryEmbedding := embedding.Embeddings[0].Embedding

	// Search in store
	results, err := s.store.Search(ctx, query, queryEmbedding, uint(limit))
	if err != nil {
		return nil, errors.Wrapf(err, "failed to search memories")
	}

	return results, nil
}

// GetMemory retrieves a memory by its ID
func (s *service) GetMemory(ctx context.Context, key string) (*Memory, error) {
	if key == "" {
		return nil, errors.Errorf("memory ID cannot be empty")
	}

	memory, err := s.store.Get(ctx, key)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get memory")
	}

	return memory, nil
}

// DeleteMemory deletes a memory by its ID
func (s *service) DeleteMemory(ctx context.Context, key string) error {
	if key == "" {
		return errors.Errorf("memory key cannot be empty")
	}

	if err := s.store.Delete(ctx, key); err != nil {
		return errors.Wrapf(err, "failed to delete memory")
	}

	return nil
}

// ListMemories returns all stored memories
func (s *service) ListMemories(ctx context.Context) ([]*Memory, error) {
	memories, err := s.store.List(ctx)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to list memories")
	}

	return memories, nil
}

func (s *service) UpdateMemory(ctx context.Context, key string, input UpdateMemoryInput) (*Memory, error) {
	memory, err := s.GetMemory(ctx, key)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get memory")
	}

	if input.Value != nil {
		memory.Value = *input.Value
		// Generate embedding for the input
		embedding, err := s.embedder.Embed(ctx, &ai.EmbedRequest{
			Input: []*ai.Document{{Content: []*ai.Part{ai.NewTextPart(memory.Value)}}},
		})
		if err != nil {
			return nil, errors.Wrapf(err, "failed to generate embedding for input '%s'", memory.Value)
		}
		memory.Embedding = embedding.Embeddings[0].Embedding
	}

	if len(input.Tags) > 0 {
		memory.Tags = input.Tags
	}

	if err := s.store.Replace(ctx, memory); err != nil {
		return nil, errors.Wrapf(err, "failed to update memory")
	}

	return memory, nil
}
