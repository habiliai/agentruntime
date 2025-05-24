package openaiapi

import (
	"context"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	goopenai "github.com/openai/openai-go"
)

func DefineModel(g *genkit.Genkit, client *goopenai.Client, labelPrefix, provider, name string, caps ai.ModelSupports) ai.Model {
	meta := &ai.ModelInfo{
		Label:    labelPrefix + " - " + name,
		Supports: &caps,
	}
	return genkit.DefineModel(
		g,
		provider,
		name,
		meta,
		func(ctx context.Context, req *ai.ModelRequest, _ core.StreamCallback[*ai.ModelResponseChunk]) (*ai.ModelResponse, error) {
			return generate(ctx, client, name, req)
		},
	)
}

func DefineEmbedder(g *genkit.Genkit, client *goopenai.Client, provider, name string) ai.Embedder {
	return genkit.DefineEmbedder(g, provider, name, func(ctx context.Context, input *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		var data goopenai.EmbeddingNewParamsInputArrayOfStrings
		for _, doc := range input.Input {
			for _, p := range doc.Content {
				data = append(data, p.Text)
			}
		}

		params := goopenai.EmbeddingNewParams{
			Input:          goopenai.F[goopenai.EmbeddingNewParamsInputUnion](data),
			Model:          goopenai.F(name),
			EncodingFormat: goopenai.F(goopenai.EmbeddingNewParamsEncodingFormatFloat),
		}

		embRes, err := client.Embeddings.New(ctx, params)
		if err != nil {
			return nil, err
		}

		var res ai.EmbedResponse
		for _, emb := range embRes.Data {
			embedding := make([]float32, len(emb.Embedding))
			for i, val := range emb.Embedding {
				embedding[i] = float32(val)
			}
			res.Embeddings = append(res.Embeddings, &ai.Embedding{Embedding: embedding})
		}
		return &res, nil
	})
}

func generate(
	ctx context.Context,
	client *goopenai.Client,
	model string,
	input *ai.ModelRequest,
) (*ai.ModelResponse, error) {
	req, err := convertRequest(model, input)
	if err != nil {
		return nil, err
	}

	res, err := client.Chat.Completions.New(ctx, req)
	if err != nil {
		return nil, err
	}

	jsonMode := false
	if input.Output != nil &&
		input.Output.Format == ai.OutputFormatJSON {
		jsonMode = true
	}

	r := translateResponse(res, jsonMode)
	r.Request = input
	return r, nil
}
