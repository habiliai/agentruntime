package knowledge_test

import (
	"bytes"
	"fmt"
	"io"
	"log/slog"
	"os"
	"testing"

	"github.com/habiliai/agentruntime/config"
	xgenkit "github.com/habiliai/agentruntime/internal/genkit"
	"github.com/habiliai/agentruntime/knowledge"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test data constants
const (
	testCSVData = `name,age,city
John,25,New York
Alice,30,London
Bob,35,Tokyo`

	testJSONArrayData = `[
	{"name": "John", "age": 25, "city": "New York", "description": "Software engineer from NY"},
	{"name": "Alice", "age": 30, "city": "London", "description": "Designer from London"},
	{"name": "Bob", "age": 35, "city": "Tokyo", "description": "Manager from Tokyo"}
]`

	testJSONObjectData = `{
	"title": "Company Information",
	"description": "This is information about our company",
	"employees": 150,
	"location": "San Francisco",
	"founded": 2020
}`

	testTextData = `This is a sample text document for testing.

It contains multiple paragraphs with different content.
Each paragraph should be processed separately to create meaningful chunks.

The last paragraph contains some technical information about AI and machine learning
to test the embedding and search functionality.`

	testMarkdownData = `# Main Title

This is the introduction section of the document.

## Section 1: Overview

This section provides an overview of the system.
It includes technical details and implementation notes.

### Subsection 1.1: Architecture

The architecture consists of multiple components:
- Component A
- Component B  
- Component C

## Section 2: Implementation

This section covers implementation details.

### Subsection 2.1: Database

Database configuration and setup information.

### Subsection 2.2: API

API endpoints and documentation.`
)

func TestIndexKnowledgeFromDocuments(t *testing.T) {
	ctx := t.Context()

	// Skip if no API keys
	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" && !testing.Short() {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	// Initialize service
	modelConfig := &config.ModelConfig{
		OpenAIAPIKey: "test-key",
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	knowledgeConfig := config.NewKnowledgeConfig()
	service, err := knowledge.NewService(ctx, modelConfig, knowledgeConfig, logger)
	require.NoError(t, err)
	defer service.Close()

	// Create document readers for different types
	documents := func(yield func(knowledge.DocumentReader, error) bool) {
		// CSV document
		if !yield(knowledge.DocumentReader{
			Content:     bytes.NewReader([]byte(testCSVData)),
			ContentType: "text/csv",
		}, nil) {
			return
		}

		// JSON array document
		if !yield(knowledge.DocumentReader{
			Content:     bytes.NewReader([]byte(testJSONArrayData)),
			ContentType: "application/json",
		}, nil) {
			return
		}

		// Text document
		if !yield(knowledge.DocumentReader{
			Content:     bytes.NewReader([]byte(testTextData)),
			ContentType: "text/plain",
		}, nil) {
			return
		}

		// Markdown document
		yield(knowledge.DocumentReader{
			Content:     bytes.NewReader([]byte(testMarkdownData)),
			ContentType: "text/markdown",
		}, nil)
	}

	// Process documents
	result, err := service.IndexKnowledgeFromDocuments(ctx, "test-multi-doc", documents)

	// If no API key, expect error
	if nomicApiKey == "" {
		require.Error(t, err)
		t.Logf("Expected error (no API key): %v", err)
		return
	}

	require.NoError(t, err)
	require.NotNil(t, result)

	// Validate basic structure
	assert.Equal(t, "test-multi-doc", result.ID)
	assert.Equal(t, knowledge.SourceTypeDocument, result.Metadata[knowledge.MetadataKeySourceType])
	assert.Equal(t, 4, result.Metadata["document_count"]) // CSV, JSON, Text, Markdown
	assert.NotEmpty(t, result.Documents)

	// Check document types in metadata
	documentTypes, ok := result.Metadata["document_types"].([]string)
	require.True(t, ok)
	assert.Contains(t, fmt.Sprintf("%v", documentTypes), "csv")
	assert.Contains(t, fmt.Sprintf("%v", documentTypes), "json")
	assert.Contains(t, fmt.Sprintf("%v", documentTypes), "text")
	assert.Contains(t, fmt.Sprintf("%v", documentTypes), "markdown")

	t.Logf("Processed %d documents into %d chunks", result.Metadata["document_count"], len(result.Documents))
	t.Logf("Document types: %v", documentTypes)

	// Validate that all documents have embeddings
	for i, doc := range result.Documents {
		assert.NotEmpty(t, doc.Embeddings, "document %d should have embeddings", i)
		assert.NotEmpty(t, doc.EmbeddingText, "document %d should have embedding text", i)
		assert.NotEmpty(t, doc.Content.MIMEType, "document %d should have MIME type", i)
		assert.NotEmpty(t, doc.ID, "document %d should have ID", i)

		// Check metadata
		assert.NotNil(t, doc.Metadata["document_number"], "document %d should have document_number", i)
		assert.NotNil(t, doc.Metadata["global_index"], "document %d should have global_index", i)
		assert.NotNil(t, doc.Metadata["source_content_type"], "document %d should have source_content_type", i)
		assert.NotNil(t, doc.Metadata["source_type"], "document %d should have source_type", i)
	}

	// Test that we can retrieve knowledge
	retrieved, err := service.GetKnowledge(ctx, "test-multi-doc")
	require.NoError(t, err)
	assert.Equal(t, result.ID, retrieved.ID)
	assert.Equal(t, len(result.Documents), len(retrieved.Documents))
}

func TestProcessKnowledgeFromMultipleDocuments(t *testing.T) {
	ctx := t.Context()

	// Skip if no API keys
	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" && !testing.Short() {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	// Initialize genkit and embedder
	modelConfig := &config.ModelConfig{
		OpenAIAPIKey: "test-key",
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	g, err := xgenkit.NewGenkit(ctx, modelConfig, logger, false)
	require.NoError(t, err)

	embedder := knowledge.NewEmbedder(nomicApiKey)
	knowledgeConfig := config.NewKnowledgeConfig()

	// Create document readers
	documents := func(yield func(knowledge.DocumentReader, error) bool) {
		if !yield(knowledge.DocumentReader{
			Content:     bytes.NewReader([]byte(testCSVData)),
			ContentType: "text/csv",
		}, nil) {
			return
		}

		yield(knowledge.DocumentReader{
			Content:     bytes.NewReader([]byte(testJSONArrayData)),
			ContentType: "application/json",
		}, nil)
	}

	// Process documents
	result, err := knowledge.ProcessKnowledgeFromMultipleDocuments(ctx, g, "test-mixed", documents, logger, knowledgeConfig, embedder)

	if nomicApiKey == "" {
		require.Error(t, err)
		return
	}

	require.NoError(t, err)
	require.NotNil(t, result)

	// Validate structure
	assert.Equal(t, "test-mixed", result.ID)
	assert.Equal(t, 2, result.Metadata["document_count"])
	assert.NotEmpty(t, result.Documents)

	t.Logf("Mixed document processing result: %d total chunks", len(result.Documents))
}

func TestProcessDocumentsFromCSV(t *testing.T) {
	ctx := t.Context()

	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" && !testing.Short() {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	embedder := knowledge.NewEmbedder(nomicApiKey)

	reader := bytes.NewReader([]byte(testCSVData))
	documents, metadata, err := knowledge.ProcessDocumentsFromCSV(ctx, reader, logger, embedder, 1)

	if nomicApiKey == "" {
		require.Error(t, err)
		return
	}

	require.NoError(t, err)
	require.NotNil(t, documents)
	require.NotNil(t, metadata)

	// Should have 3 rows (excluding header)
	assert.Equal(t, 3, len(documents))
	assert.Equal(t, knowledge.SourceTypeCSV, metadata["source_type"])
	assert.Equal(t, 3, metadata["total_rows"])

	// Check first document
	doc := documents[0]
	assert.Contains(t, doc.EmbeddingText, "John")
	assert.Contains(t, doc.EmbeddingText, "25")
	assert.Contains(t, doc.EmbeddingText, "New York")
	assert.Equal(t, "text/plain", doc.Content.MIMEType)
	assert.Equal(t, 1, doc.Metadata["row_number"])
	assert.NotEmpty(t, doc.Embeddings)

	t.Logf("CSV processed: %d rows", len(documents))
}

func TestProcessDocumentsFromJSON(t *testing.T) {
	ctx := t.Context()

	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" && !testing.Short() {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	embedder := knowledge.NewEmbedder(nomicApiKey)

	tests := []struct {
		name     string
		data     string
		expected int
	}{
		{
			name:     "JSON array",
			data:     testJSONArrayData,
			expected: 3, // 3 objects in array
		},
		{
			name:     "JSON object",
			data:     testJSONObjectData,
			expected: 1, // 1 object
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := bytes.NewReader([]byte(tt.data))
			documents, metadata, err := knowledge.ProcessDocumentsFromJSON(ctx, reader, logger, embedder, 1)

			if nomicApiKey == "" {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, documents)
			require.NotNil(t, metadata)

			assert.Equal(t, tt.expected, len(documents))
			assert.Equal(t, knowledge.SourceTypeJSON, metadata["source_type"])

			// Check that all documents have embeddings
			for i, doc := range documents {
				assert.NotEmpty(t, doc.Embeddings, "document %d should have embeddings", i)
				assert.NotEmpty(t, doc.EmbeddingText, "document %d should have text", i)
				assert.Equal(t, "text/plain", doc.Content.MIMEType)
			}

			t.Logf("JSON %s processed: %d documents", tt.name, len(documents))
		})
	}
}

func TestProcessDocumentsFromText(t *testing.T) {
	ctx := t.Context()

	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" && !testing.Short() {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	embedder := knowledge.NewEmbedder(nomicApiKey)

	reader := bytes.NewReader([]byte(testTextData))
	documents, metadata, err := knowledge.ProcessDocumentsFromText(ctx, reader, logger, embedder, 1)

	if nomicApiKey == "" {
		require.Error(t, err)
		return
	}

	require.NoError(t, err)
	require.NotNil(t, documents)
	require.NotNil(t, metadata)

	assert.Greater(t, len(documents), 0)
	assert.Equal(t, knowledge.SourceTypeText, metadata["source_type"])

	// Check that all documents have embeddings and content
	for i, doc := range documents {
		assert.NotEmpty(t, doc.Embeddings, "document %d should have embeddings", i)
		assert.NotEmpty(t, doc.EmbeddingText, "document %d should have text", i)
		assert.Equal(t, "text/plain", doc.Content.MIMEType)
		assert.NotNil(t, doc.Metadata["chunk_number"])
	}

	t.Logf("Text processed: %d chunks", len(documents))
}

func TestProcessDocumentsFromMarkdown(t *testing.T) {
	ctx := t.Context()

	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" && !testing.Short() {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	embedder := knowledge.NewEmbedder(nomicApiKey)

	reader := bytes.NewReader([]byte(testMarkdownData))
	documents, metadata, err := knowledge.ProcessDocumentsFromMarkdown(ctx, reader, logger, embedder, 1)

	if nomicApiKey == "" {
		require.Error(t, err)
		return
	}

	require.NoError(t, err)
	require.NotNil(t, documents)
	require.NotNil(t, metadata)

	assert.Greater(t, len(documents), 0)
	assert.Equal(t, knowledge.SourceTypeMarkdown, metadata["source_type"])

	// Check that sections are properly split
	foundMainTitle := false
	foundSection1 := false
	foundSection2 := false

	for i, doc := range documents {
		assert.NotEmpty(t, doc.Embeddings, "document %d should have embeddings", i)
		assert.NotEmpty(t, doc.EmbeddingText, "document %d should have text", i)
		assert.Equal(t, "text/markdown", doc.Content.MIMEType)
		assert.NotNil(t, doc.Metadata["section_number"])

		// Check for section content
		text := doc.EmbeddingText
		if bytes.Contains([]byte(text), []byte("Main Title")) {
			foundMainTitle = true
		}
		if bytes.Contains([]byte(text), []byte("Section 1: Overview")) {
			foundSection1 = true
		}
		if bytes.Contains([]byte(text), []byte("Section 2: Implementation")) {
			foundSection2 = true
		}
	}

	assert.True(t, foundMainTitle, "Should find main title section")
	assert.True(t, foundSection1, "Should find section 1")
	assert.True(t, foundSection2, "Should find section 2")

	t.Logf("Markdown processed: %d sections", len(documents))
}

func TestProcessDocumentsByType_UnknownType(t *testing.T) {
	ctx := t.Context()

	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" && !testing.Short() {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	// Initialize genkit and embedder
	modelConfig := &config.ModelConfig{
		OpenAIAPIKey: "test-key",
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	g, err := xgenkit.NewGenkit(ctx, modelConfig, logger, false)
	require.NoError(t, err)

	embedder := knowledge.NewEmbedder(nomicApiKey)
	knowledgeConfig := config.NewKnowledgeConfig()

	// Test with unknown content type
	docReader := knowledge.DocumentReader{
		Content:     bytes.NewReader([]byte("This is some text content")),
		ContentType: "application/unknown",
	}

	documents, metadata, err := knowledge.ProcessDocumentsByType(ctx, g, docReader, logger, knowledgeConfig, embedder, 1)

	if nomicApiKey == "" {
		require.Error(t, err)
		return
	}

	require.NoError(t, err)
	require.NotNil(t, documents)
	require.NotNil(t, metadata)

	// Should process as text
	assert.Greater(t, len(documents), 0)
	assert.Equal(t, knowledge.SourceTypeText, metadata["source_type"])

	t.Logf("Unknown type processed as text: %d chunks", len(documents))
}

func TestDocumentReader_ErrorHandling(t *testing.T) {
	ctx := t.Context()

	// Initialize service
	modelConfig := &config.ModelConfig{
		OpenAIAPIKey: "test-key",
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	knowledgeConfig := config.NewKnowledgeConfig()
	service, err := knowledge.NewService(ctx, modelConfig, knowledgeConfig, logger)
	require.NoError(t, err)
	defer service.Close()

	// Create iterator that yields an error
	documentsWithError := func(yield func(knowledge.DocumentReader, error) bool) {
		// First valid document
		if !yield(knowledge.DocumentReader{
			Content:     bytes.NewReader([]byte(testTextData)),
			ContentType: "text/plain",
		}, nil) {
			return
		}

		// Document with error
		yield(knowledge.DocumentReader{}, fmt.Errorf("test error"))
	}

	// Process should fail due to error
	_, err = service.IndexKnowledgeFromDocuments(ctx, "test-error", documentsWithError)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "test error")

	t.Logf("Error handling works correctly: %v", err)
}

// Integration test with real search
func TestIndexKnowledgeFromDocuments_WithSearch(t *testing.T) {
	ctx := t.Context()

	// Skip if no API keys
	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" && !testing.Short() {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	// Initialize service
	modelConfig := &config.ModelConfig{
		OpenAIAPIKey: "test-key",
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	knowledgeConfig := config.NewKnowledgeConfig()
	service, err := knowledge.NewService(ctx, modelConfig, knowledgeConfig, logger)
	require.NoError(t, err)
	defer service.Close()

	// Create diverse documents with searchable content
	documents := func(yield func(knowledge.DocumentReader, error) bool) {
		// CSV with employee data
		employeeCSV := `name,role,department,description
John Smith,Engineer,Development,Experienced software engineer specializing in backend systems
Alice Johnson,Designer,UX,Creative designer with expertise in user experience and interfaces  
Bob Wilson,Manager,Operations,Operations manager handling logistics and supply chain`

		if !yield(knowledge.DocumentReader{
			Content:     bytes.NewReader([]byte(employeeCSV)),
			ContentType: "text/csv",
		}, nil) {
			return
		}

		// JSON with product data
		productJSON := `[
	{"name": "Widget Pro", "category": "electronics", "description": "Advanced electronic widget with smart features"},
	{"name": "Gadget Max", "category": "electronics", "description": "High-performance gadget for professional use"},
	{"name": "Tool Kit", "category": "hardware", "description": "Complete toolkit for mechanical repairs"}
]`

		if !yield(knowledge.DocumentReader{
			Content:     bytes.NewReader([]byte(productJSON)),
			ContentType: "application/json",
		}, nil) {
			return
		}

		// Technical documentation
		techDoc := `# System Architecture

Our system uses a microservices architecture with the following components:

## Backend Services
- Authentication service handles user login and security
- Data processing service manages large datasets
- API gateway routes requests to appropriate services

## Frontend Applications  
- Web application built with React
- Mobile app for iOS and Android
- Admin dashboard for system management`

		yield(knowledge.DocumentReader{
			Content:     bytes.NewReader([]byte(techDoc)),
			ContentType: "text/markdown",
		}, nil)
	}

	// Process documents
	knowledge, err := service.IndexKnowledgeFromDocuments(ctx, "test-search", documents)

	if nomicApiKey == "" {
		require.Error(t, err)
		return
	}

	require.NoError(t, err)
	require.NotNil(t, knowledge)

	// Test search functionality
	searchTests := []struct {
		query    string
		expected []string // Expected terms to find in results
	}{
		{
			query:    "software engineer",
			expected: []string{"John", "engineer", "backend"},
		},
		{
			query:    "electronics widget",
			expected: []string{"Widget", "electronic", "smart"},
		},
		{
			query:    "microservices architecture",
			expected: []string{"microservices", "architecture", "services"},
		},
		{
			query:    "user experience design",
			expected: []string{"Alice", "designer", "user"},
		},
	}

	for _, tt := range searchTests {
		t.Run(fmt.Sprintf("search_%s", tt.query), func(t *testing.T) {
			results, err := service.RetrieveRelevantKnowledge(ctx, tt.query, 5, []string{"test-search"})
			require.NoError(t, err)
			require.NotEmpty(t, results, "should find results for query: %s", tt.query)

			// Check that results contain expected terms
			foundTerms := make(map[string]bool)
			for _, result := range results {
				text := result.EmbeddingText
				for _, term := range tt.expected {
					if bytes.Contains([]byte(text), []byte(term)) {
						foundTerms[term] = true
					}
				}
			}

			// Should find at least one expected term
			assert.Greater(t, len(foundTerms), 0,
				"should find at least one expected term for query '%s', found terms: %v",
				tt.query, foundTerms)

			t.Logf("Query '%s' found %d results with terms: %v",
				tt.query, len(results), foundTerms)
		})
	}
}
