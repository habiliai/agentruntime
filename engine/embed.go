package engine

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/pkg/errors"
)

func (e *Engine) Embed(
	ctx context.Context,
	texts ...string,
) ([][]float32, error) {
	embedder := genkit.LookupEmbedder(e.genkit, "openai/text-embedding-3-small")
	if embedder == nil {
		return nil, errors.New("embedder not found")
	}

	resp, err := genkit.Embed(ctx, e.genkit, ai.WithTextDocs(texts...), ai.WithEmbedder(embedder))
	if err != nil {
		return nil, err
	}

	embeddings := make([][]float32, len(resp.Embeddings))
	for i, embedding := range resp.Embeddings {
		embeddings[i] = embedding.Embedding
	}

	return embeddings, nil
}
