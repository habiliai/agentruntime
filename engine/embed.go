package engine

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func (e *engine) Embed(
	ctx context.Context,
	texts ...string,
) ([][]float32, error) {
	embedder := genkit.LookupEmbedder(e.genkit, "openai", "text-embedding-3-small")

	resp, err := ai.Embed(ctx, embedder, ai.WithTextDocs(texts...))
	if err != nil {
		return nil, err
	}

	embeddings := make([][]float32, len(resp.Embeddings))
	for i, embedding := range resp.Embeddings {
		embeddings[i] = embedding.Embedding
	}

	return embeddings, nil
}
