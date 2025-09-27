package anthropic

import (
	"context"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModel(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Anthropic{
		APIKey: "test-key",
	}))

	tests := []struct {
		name      string
		modelName string
		wantNil   bool
	}{
		{
			name:      "claude-4-opus exists",
			modelName: "claude-4-opus",
			wantNil:   false,
		},
		{
			name:      "claude-4-sonnet exists",
			modelName: "claude-4-sonnet",
			wantNil:   false,
		},
		{
			name:      "claude-3.7-sonnet exists",
			modelName: "claude-3.7-sonnet",
			wantNil:   false,
		},
		{
			name:      "unknown model",
			modelName: "claude-2-legacy",
			wantNil:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := Model(g, tt.modelName)
			if tt.wantNil {
				assert.Nil(t, model)
			} else {
				assert.NotNil(t, model)
			}
		})
	}
}

func TestKnownModels(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Anthropic{
		APIKey: "test-key",
	}))

	// Test that known models are registered with correct capabilities
	opus := Model(g, "claude-4-opus")
	require.NotNil(t, opus)

	sonnet := Model(g, "claude-4-sonnet")
	require.NotNil(t, sonnet)

	sonnet37 := Model(g, "claude-3.7-sonnet")
	require.NotNil(t, sonnet37)

	haiku := Model(g, "claude-3.5-haiku")
	require.NotNil(t, haiku)
}
