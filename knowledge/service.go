package knowledge

import (
	"context"
	"io"
	"iter"
	"log/slog"
	"sort"

	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/config"
	xgenkit "github.com/habiliai/agentruntime/internal/genkit"
	"github.com/pkg/errors"
)

type (
	DocumentReader struct {
		Content     io.Reader
		ContentType string // text/plain, text/markdown, text/csv, text/json, application/pdf
	}

	Service interface {
		// Knowledge management methods
		IndexKnowledgeFromMap(ctx context.Context, id string, input []map[string]any) (*Knowledge, error)
		IndexKnowledgeFromDocuments(ctx context.Context, id string, inputs iter.Seq2[*DocumentReader, error]) (*Knowledge, error)
		RetrieveRelevantKnowledge(ctx context.Context, query string, limit int, allowedKnowledgeIds []string) ([]*KnowledgeSearchResult, error)
		DeleteKnowledge(ctx context.Context, knowledgeId string) error
		Close() error
		GetKnowledge(ctx context.Context, knowledgeId string) (*Knowledge, error)
	}

	service struct {
		genkit *genkit.Genkit

		store         Store
		embedder      Embedder
		reranker      Reranker
		queryRewriter QueryRewriter
		config        *config.KnowledgeConfig
		logger        *slog.Logger
	}
)

var (
	_ Service = (*service)(nil)
)

// NewService creates a new knowledge service with default SQLite-based storage
func NewService(ctx context.Context, modelConfig *config.ModelConfig, conf *config.KnowledgeConfig, logger *slog.Logger) (Service, error) {
	return NewServiceWithStore(ctx, conf, modelConfig, logger, NewInMemoryStore())
}

// NewServiceWithStore creates a new knowledge service with a custom knowledge store
func NewServiceWithStore(
	ctx context.Context,
	conf *config.KnowledgeConfig,
	modelConfig *config.ModelConfig,
	logger *slog.Logger,
	store Store,
) (Service, error) {
	genkit, err := xgenkit.NewGenkit(ctx, modelConfig, logger, modelConfig.TraceVerbose)
	if err != nil {
		return nil, err
	}

	// Create embedder for RAG functionality
	embedder := NewEmbedder(conf.NomicAPIKey)

	// Create reranker if enabled
	var reranker Reranker
	if conf.RerankEnabled {
		if conf.UseBatchRerank {
			reranker = NewBatchGenkitReranker(genkit, conf.RerankModel)
		} else {
			reranker = NewGenkitReranker(genkit, conf.RerankModel)
		}
	} else {
		reranker = NewNoOpReranker()
	}

	// Create query rewriter if enabled
	var queryRewriter QueryRewriter
	if conf.QueryRewriteEnabled {
		model := conf.QueryRewriteModel
		if model == "" {
			model = conf.RerankModel // Default to rerank model
		}
		queryRewriter = CreateQueryRewriter(genkit, conf.QueryRewriteStrategy, model)
	} else {
		queryRewriter = NewNoOpQueryRewriter()
	}

	return &service{
		genkit:        genkit,
		store:         store,
		embedder:      embedder,
		reranker:      reranker,
		queryRewriter: queryRewriter,
		config:        conf,
		logger:        logger,
	}, nil
}

func (s *service) GetKnowledge(ctx context.Context, knowledgeId string) (*Knowledge, error) {
	return s.store.GetKnowledgeById(ctx, knowledgeId)
}

func (s *service) Close() error {
	if s.store != nil {
		return s.store.Close()
	}
	return nil
}

// RetrieveRelevantKnowledge retrieves relevant knowledge chunks based on query
func (s *service) RetrieveRelevantKnowledge(ctx context.Context, query string, limit int, allowedKnowledgeIds []string) ([]*KnowledgeSearchResult, error) {
	// Apply query rewriting
	queries, err := s.queryRewriter.Rewrite(ctx, query)
	if err != nil {
		// Log error but continue with original query
		s.logger.Warn("query rewriting failed, using original query", slog.String("error", err.Error()))
		queries = []string{query}
	}

	// Determine retrieval count based on rerank configuration
	retrievalLimit := limit
	if s.config.RerankEnabled && s.config.RetrievalFactor > 1 {
		retrievalLimit = limit * s.config.RetrievalFactor
	}

	// Search with all rewritten queries
	allSearchResults := make([]KnowledgeSearchResult, 0)
	uniqueResults := make(map[string]KnowledgeSearchResult) // Use map to track unique results by ID

	for i, q := range queries {
		// Generate embedding for this query
		embeddings, err := s.embedder.EmbedTexts(ctx, EmbeddingTaskTypeQuery, q)
		if err != nil {
			s.logger.Warn("failed to generate embedding for rewritten query",
				slog.String("query", q),
				slog.String("error", err.Error()))
			continue // Skip this query rather than using empty embedding
		}

		if len(embeddings) == 0 {
			continue // Skip if no embeddings returned
		}

		queryEmbedding := embeddings[0]

		// Search for relevant knowledge
		searchResults, err := s.store.Search(ctx, queryEmbedding, retrievalLimit, allowedKnowledgeIds)
		if err != nil {
			s.logger.Warn("search failed for rewritten query",
				slog.String("query", q),
				slog.String("error", err.Error()))
			continue
		}

		// Apply score weighting based on query type
		scoreWeight := 1.0
		if i > 0 { // Not the original query
			scoreWeight = 0.9 // Slightly lower weight for rewritten queries
		}

		// Merge results, keeping highest score for duplicates
		for _, result := range searchResults {
			adjustedScore := result.Score * float32(scoreWeight)
			if existing, exists := uniqueResults[result.ID]; !exists || adjustedScore > existing.Score {
				result.Score = adjustedScore
				uniqueResults[result.ID] = result
			}
		}
	}

	// Convert map back to slice
	for _, result := range uniqueResults {
		allSearchResults = append(allSearchResults, result)
	}

	// Sort by score descending
	sort.Slice(allSearchResults, func(i, j int) bool {
		return allSearchResults[i].Score > allSearchResults[j].Score
	})

	// Extract content for reranking
	candidates := make([]*KnowledgeSearchResult, len(allSearchResults))
	for i, result := range allSearchResults {
		candidates[i] = &result
	}

	// Apply reranking if enabled
	if s.config.RerankEnabled && s.reranker != nil && len(candidates) > limit {
		rerankResults, err := s.reranker.Rerank(ctx, query, candidates, limit)
		if err != nil {
			// If reranking fails, fall back to original results
			s.logger.Warn("reranking failed, falling back to original results", slog.String("error", err.Error()))
			if len(candidates) > limit {
				candidates = candidates[:limit]
			}
			return candidates, nil
		}

		return rerankResults, nil
	}

	// If reranking is not enabled or not needed, return original results
	if len(candidates) > limit {
		candidates = candidates[:limit]
	}
	return candidates, nil
}

// IndexKnowledgeFromDocuments processes multiple documents with different types and merges them into a single Knowledge object
func (s *service) IndexKnowledgeFromDocuments(ctx context.Context, id string, inputs iter.Seq2[*DocumentReader, error]) (*Knowledge, error) {
	// First, delete existing knowledge for this ID
	if id != "" {
		if err := s.DeleteKnowledge(ctx, id); err != nil {
			return nil, errors.Wrapf(err, "failed to delete existing knowledge")
		}
	}

	knowledge, err := ProcessKnowledgeFromMultipleDocuments(ctx, s.genkit, id, inputs, s.logger, s.config, s.embedder)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to process knowledge from documents")
	}

	// Store all items
	if err := s.store.Store(ctx, knowledge); err != nil {
		return nil, errors.Wrapf(err, "failed to store knowledge")
	}

	return knowledge, nil
}

// DeleteAgentKnowledge removes all knowledge for an agent
func (s *service) DeleteKnowledge(ctx context.Context, knowledgeId string) error {
	return s.store.DeleteKnowledgeById(ctx, knowledgeId)
}
