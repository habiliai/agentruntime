package memory_test

import (
	"fmt"
	"log/slog"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/config"
	internalgenkit "github.com/habiliai/agentruntime/internal/genkit"
	"github.com/habiliai/agentruntime/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInMemoryStore_Set(t *testing.T) {
	store := memory.NewInMemoryStore()
	ctx := t.Context()

	mem := &memory.Memory{
		Key:       "test-key",
		Value:     "test value",
		Source:    memory.MemorySourceUser,
		Tags:      []string{"tag1", "tag2"},
		Embedding: []float32{0.1, 0.2, 0.3},
	}

	err := store.Set(ctx, mem)
	require.NoError(t, err, "Set should not return an error")

	// Verify the memory was stored
	stored, err := store.Get(ctx, "test-key")
	require.NoError(t, err)
	assert.Equal(t, mem.Key, stored.Key)
	assert.Equal(t, mem.Value, stored.Value)
	assert.Equal(t, mem.Source, stored.Source)
	assert.Equal(t, mem.Tags, stored.Tags)
	assert.Equal(t, mem.Embedding, stored.Embedding)
}

func TestInMemoryStore_Get(t *testing.T) {
	store := memory.NewInMemoryStore()
	ctx := t.Context()

	// Test getting non-existent memory
	_, err := store.Get(ctx, "non-existent")
	assert.Error(t, err, "Get should return error for non-existent key")
	assert.Contains(t, err.Error(), "not found")

	// Set a memory first
	mem := &memory.Memory{
		Key:       "existing-key",
		Value:     "existing value",
		Source:    memory.MemorySourceAgent,
		Embedding: []float32{0.4, 0.5, 0.6},
	}

	err = store.Set(ctx, mem)
	require.NoError(t, err)

	// Test getting existing memory
	retrieved, err := store.Get(ctx, "existing-key")
	require.NoError(t, err, "Get should not return error for existing key")
	assert.Equal(t, mem.Key, retrieved.Key)
	assert.Equal(t, mem.Value, retrieved.Value)
	assert.Equal(t, mem.Source, retrieved.Source)
}

func TestInMemoryStore_Search(t *testing.T) {
	store := memory.NewInMemoryStore()
	ctx := t.Context()

	// Test search with empty store
	_, err := store.Search(ctx, "test query", []float32{0.1, 0.2, 0.3}, 10)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no memories found")

	// Test search with empty embedding
	_, err = store.Search(ctx, "test query", []float32{}, 10)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "query embedding is empty")

	// Add some test memories with embeddings
	memories := []*memory.Memory{
		{
			Key:       "mem1",
			Value:     "first memory",
			Source:    memory.MemorySourceUser,
			Embedding: []float32{1.0, 0.0, 0.0}, // Very similar to query
		},
		{
			Key:       "mem2",
			Value:     "second memory",
			Source:    memory.MemorySourceAgent,
			Embedding: []float32{0.0, 1.0, 0.0}, // Less similar to query
		},
		{
			Key:       "mem3",
			Value:     "third memory",
			Source:    memory.MemorySourceUser,
			Embedding: []float32{-1.0, 0.0, 0.0}, // Least similar to query
		},
		{
			Key:       "mem4",
			Value:     "fourth memory with different dimensions",
			Source:    memory.MemorySourceAgent,
			Embedding: []float32{0.1, 0.2}, // Different dimensions - should be excluded
		},
	}

	// Set all memories
	for _, mem := range memories {
		err := store.Set(ctx, mem)
		require.NoError(t, err)
	}

	// Test search with matching dimensions
	queryEmbedding := []float32{0.9, 0.1, 0.1}
	results, err := store.Search(ctx, "test query", queryEmbedding, 10)
	require.NoError(t, err, "Search should not return error")

	// Should only return memories with matching embedding dimensions (3)
	assert.Len(t, results, 3, "Should return 3 memories with matching dimensions")

	// Results should be sorted by similarity score (descending)
	assert.True(t, results[0].Score >= results[1].Score, "Results should be sorted by score")
	assert.True(t, results[1].Score >= results[2].Score, "Results should be sorted by score")

	// The most similar should be mem1 (highest score)
	assert.Equal(t, "mem1", results[0].Memory.Key)

	// Test limit parameter
	limitedResults, err := store.Search(ctx, "test query", queryEmbedding, 2)
	require.NoError(t, err)
	assert.Len(t, limitedResults, 2, "Should respect limit parameter")

	// Test limit of 0 (should return all)
	allResults, err := store.Search(ctx, "test query", queryEmbedding, 0)
	require.NoError(t, err)
	assert.Len(t, allResults, 3, "Limit of 0 should return all results")
}

func TestInMemoryStore_List(t *testing.T) {
	store := memory.NewInMemoryStore()
	ctx := t.Context()

	// Test empty store
	memories, err := store.List(ctx)
	require.NoError(t, err, "List should not return error")
	assert.Empty(t, memories, "Empty store should return empty list")

	// Add some memories
	testMemories := []*memory.Memory{
		{
			Key:    "key1",
			Value:  "value1",
			Source: memory.MemorySourceUser,
			Tags:   []string{"tag1"},
		},
		{
			Key:    "key2",
			Value:  "value2",
			Source: memory.MemorySourceAgent,
			Tags:   []string{"tag2", "tag3"},
		},
	}

	for _, mem := range testMemories {
		err := store.Set(ctx, mem)
		require.NoError(t, err)
	}

	// Test list with memories
	memories, err = store.List(ctx)
	require.NoError(t, err, "List should not return error")
	assert.Len(t, memories, 2, "Should return all stored memories")

	// Verify all memories are returned (order not guaranteed)
	keys := make(map[string]bool)
	for _, mem := range memories {
		keys[mem.Key] = true
	}
	assert.True(t, keys["key1"], "Should contain key1")
	assert.True(t, keys["key2"], "Should contain key2")
}

func TestInMemoryStore_Delete(t *testing.T) {
	store := memory.NewInMemoryStore()
	ctx := t.Context()

	// Test deleting non-existent key (should not error)
	err := store.Delete(ctx, "non-existent")
	assert.NoError(t, err, "Delete should not error on non-existent key")

	// Add a memory
	mem := &memory.Memory{
		Key:    "delete-me",
		Value:  "delete this value",
		Source: memory.MemorySourceUser,
	}

	err = store.Set(ctx, mem)
	require.NoError(t, err)

	// Verify it exists
	_, err = store.Get(ctx, "delete-me")
	require.NoError(t, err, "Memory should exist before deletion")

	// Delete it
	err = store.Delete(ctx, "delete-me")
	require.NoError(t, err, "Delete should not return error")

	// Verify it's gone
	_, err = store.Get(ctx, "delete-me")
	assert.Error(t, err, "Memory should not exist after deletion")
	assert.Contains(t, err.Error(), "not found")
}

func TestInMemoryStore_ConcurrentAccess(t *testing.T) {
	store := memory.NewInMemoryStore()
	ctx := t.Context()

	// Test concurrent access to ensure thread safety
	done := make(chan bool, 2)

	// Goroutine 1: Set memories
	go func() {
		defer func() { done <- true }()
		for i := 0; i < 100; i++ {
			mem := &memory.Memory{
				Key:    "key-" + string(rune(i)),
				Value:  "value-" + string(rune(i)),
				Source: memory.MemorySourceUser,
			}
			err := store.Set(ctx, mem)
			assert.NoError(t, err)
		}
	}()

	// Goroutine 2: List memories
	go func() {
		defer func() { done <- true }()
		for i := 0; i < 50; i++ {
			_, err := store.List(ctx)
			assert.NoError(t, err)
		}
	}()

	// Wait for both goroutines to complete
	<-done
	<-done

	// Verify final state
	memories, err := store.List(ctx)
	require.NoError(t, err)
	assert.Len(t, memories, 100, "All memories should be stored")
}

func TestInMemoryStore_SearchScoring(t *testing.T) {
	store := memory.NewInMemoryStore()
	ctx := t.Context()

	// Test the scoring mechanism with known vectors
	memories := []*memory.Memory{
		{
			Key:       "identical",
			Value:     "identical vector",
			Source:    memory.MemorySourceUser,
			Embedding: []float32{1.0, 0.0}, // Identical to query
		},
		{
			Key:       "opposite",
			Value:     "opposite vector",
			Source:    memory.MemorySourceAgent,
			Embedding: []float32{-1.0, 0.0}, // Opposite to query
		},
		{
			Key:       "orthogonal",
			Value:     "orthogonal vector",
			Source:    memory.MemorySourceUser,
			Embedding: []float32{0.0, 1.0}, // Orthogonal to query
		},
	}

	for _, mem := range memories {
		err := store.Set(ctx, mem)
		require.NoError(t, err)
	}

	queryEmbedding := []float32{1.0, 0.0}
	results, err := store.Search(ctx, "test", queryEmbedding, 10)
	require.NoError(t, err)
	require.Len(t, results, 3)

	// Verify scoring order: identical > orthogonal > opposite
	assert.Equal(t, "identical", results[0].Memory.Key)
	assert.Equal(t, "orthogonal", results[1].Memory.Key)
	assert.Equal(t, "opposite", results[2].Memory.Key)

	// Check score ranges [0, 1]
	for _, result := range results {
		assert.GreaterOrEqual(t, result.Score, 0.0, "Score should be >= 0")
		assert.LessOrEqual(t, result.Score, 1.0, "Score should be <= 1")
	}

	// Identical vector should have highest score (close to 1.0)
	assert.Greater(t, results[0].Score, 0.9, "Identical vector should have high score")

	// Opposite vector should have lowest score (close to 0.0)
	assert.Less(t, results[2].Score, 0.1, "Opposite vector should have low score")
}

func TestInMemoryStore_SearchWithGenkitEmbeddings_Live(t *testing.T) {
	// Skip this test in short mode or if OPENAI_API_KEY is not set
	if testing.Short() {
		t.Skip("Skipping live test in short mode")
	}

	// Check if we have the required environment variable
	// This will be loaded by godotenv when running with: godotenv go test ...
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey == "" {
		t.Skip("Skipping live test: OPENAI_API_KEY not found. Run with: godotenv go test ./memory -v")
	}

	store := memory.NewInMemoryStore()
	ctx := t.Context()

	g := internalgenkit.NewGenkit(ctx, &config.ModelConfig{
		OpenAIAPIKey: openaiKey,
	}, slog.Default(), false)

	embedder := genkit.LookupEmbedder(g, "openai/text-embedding-3-small")

	// Test texts with different semantic meanings
	testTexts := []string{
		"The cat sat on the mat",
		"A feline rested on the carpet",    // Similar to first
		"The dog barked loudly",            // Different topic
		"Python is a programming language", // Completely different
		"Cats are wonderful pets",          // Somewhat related to first
	}

	// Generate embeddings using genkit
	embeddings, err := genkit.Embed(ctx, g, ai.WithEmbedder(embedder), ai.WithTextDocs(testTexts...))
	require.NoError(t, err, "Failed to generate embeddings")
	require.Len(t, embeddings.Embeddings, len(testTexts), "Should have embedding for each text")

	// Verify embedding dimensions (OpenAI text-embedding-3-small has 1536 dimensions)
	for i, embedding := range embeddings.Embeddings {
		assert.Len(t, embedding.Embedding, 1536, "OpenAI embedding should have 1536 dimensions for text %d", i)
	}

	// Store memories with real embeddings
	memories := make([]*memory.Memory, len(testTexts))
	for i, text := range testTexts {
		memories[i] = &memory.Memory{
			Key:       fmt.Sprintf("mem-%d", i),
			Value:     text,
			Source:    memory.MemorySourceUser,
			Embedding: embeddings.Embeddings[i].Embedding,
		}
		err := store.Set(ctx, memories[i])
		require.NoError(t, err, "Failed to store memory %d", i)
	}

	// Test search with a query similar to the first text
	queryText := "A cat is sitting"
	queryEmbedding, err := genkit.Embed(ctx, g, ai.WithEmbedder(embedder), ai.WithTextDocs(queryText))
	require.NoError(t, err, "Failed to generate query embedding")
	require.Len(t, queryEmbedding.Embeddings, 1, "Should have one query embedding")

	// Perform search
	results, err := store.Search(ctx, queryText, queryEmbedding.Embeddings[0].Embedding, 3)
	require.NoError(t, err, "Search should not fail")
	require.Len(t, results, 3, "Should return top 3 results")

	// Verify results are sorted by similarity (descending)
	for i := 1; i < len(results); i++ {
		assert.GreaterOrEqual(t, results[i-1].Score, results[i].Score,
			"Results should be sorted by similarity score (descending)")
	}

	// The most similar should be related to cats/sitting
	// Check if the top result is about cats (either "The cat sat on the mat" or "A feline rested on the carpet")
	topResult := results[0].Memory.Value
	assert.True(t,
		strings.Contains(strings.ToLower(topResult), "cat") ||
			strings.Contains(strings.ToLower(topResult), "feline"),
		"Top result should be about cats, got: %s", topResult)

	// Print results for manual verification
	t.Logf("Query: %s", queryText)
	for i, result := range results {
		t.Logf("Result %d (score: %.4f): %s", i+1, result.Score, result.Memory.Value)
	}

	// Test with a completely different query
	dogQueryText := "Dogs making noise"
	dogQueryEmbedding, err := genkit.Embed(ctx, g, ai.WithEmbedder(embedder), ai.WithTextDocs(dogQueryText))
	require.NoError(t, err, "Failed to generate dog query embedding")

	dogResults, err := store.Search(ctx, dogQueryText, dogQueryEmbedding.Embeddings[0].Embedding, 3)
	require.NoError(t, err, "Dog search should not fail")

	// The top result should be about dogs
	dogTopResult := dogResults[0].Memory.Value
	assert.True(t,
		strings.Contains(strings.ToLower(dogTopResult), "dog") ||
			strings.Contains(strings.ToLower(dogTopResult), "bark"),
		"Top result for dog query should be about dogs, got: %s", dogTopResult)

	t.Logf("Dog Query: %s", dogQueryText)
	for i, result := range dogResults {
		t.Logf("Dog Result %d (score: %.4f): %s", i+1, result.Score, result.Memory.Value)
	}

	// Verify that different queries return different top results
	assert.NotEqual(t, results[0].Memory.Key, dogResults[0].Memory.Key,
		"Different semantic queries should return different top results")
}
