package knowledge_test

import (
	"bytes"
	_ "embed"
	"log/slog"
	"os"
	"testing"

	"github.com/habiliai/agentruntime/config"
	xgenkit "github.com/habiliai/agentruntime/internal/genkit"
	"github.com/habiliai/agentruntime/knowledge"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	//go:embed testdata/gen_image_1.png
	testImage1PNG []byte

	//go:embed testdata/gen_image_2.png
	testImage2PNG []byte
)

// TestProcessDocumentFromImage tests processing a single image
func TestProcessDocumentFromImage(t *testing.T) {
	ctx := t.Context()

	// Initialize genkit
	modelConfig := &config.ModelConfig{}
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	g := xgenkit.NewGenkit(ctx, modelConfig, logger, false)

	// Create image reader
	imageReader := &knowledge.ImageReader{
		Content:     bytes.NewReader(testImage1PNG),
		ContentType: "image/png",
		Metadata: map[string]any{
			"source": "test",
		},
	}

	// Process image
	document, err := knowledge.ProcessDocumentFromImage(ctx, g, imageReader, logger, 1)

	require.NoError(t, err)
	require.NotNil(t, document)

	// Validate document structure
	assert.Equal(t, knowledge.ContentTypeImage, document.Content.Type())
	assert.NotEmpty(t, document.Content.Image, "Should have image data")
	assert.Equal(t, "image/jpeg", document.Content.MIMEType, "Should be JPEG")
	assert.Equal(t, "Image 1", document.EmbeddingText)

	// Check metadata
	assert.Equal(t, 1, document.Metadata["image_number"])
	assert.Equal(t, "png", document.Metadata["original_format"])
	assert.NotNil(t, document.Metadata["width"])
	assert.NotNil(t, document.Metadata["height"])

	t.Logf("Image dimensions: %dx%d", document.Metadata["width"], document.Metadata["height"])
}

// TestProcessKnowledgeFromMultipleImages tests processing multiple images
func TestProcessKnowledgeFromMultipleImages(t *testing.T) {
	ctx := t.Context()

	// Check if we have NOMIC API key
	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	// Initialize genkit
	modelConfig := &config.ModelConfig{}
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	g := xgenkit.NewGenkit(ctx, modelConfig, logger, false)

	// Create image iterator
	imageIterator := func(yield func(*knowledge.ImageReader, error) bool) {
		// First image
		if !yield(&knowledge.ImageReader{
			Content:     bytes.NewReader(testImage1PNG),
			ContentType: "image/png",
			Metadata: map[string]any{
				"timestamp":    0.0,
				"frame_number": 1,
			},
		}, nil) {
			return
		}

		// Second image
		yield(&knowledge.ImageReader{
			Content:     bytes.NewReader(testImage2PNG),
			ContentType: "image/png",
			Metadata: map[string]any{
				"timestamp":    1.0,
				"frame_number": 2,
			},
		}, nil)
	}

	// Create embedder
	embedder := knowledge.NewEmbedder(nomicApiKey)

	// Process multiple images
	knowledgeResult, err := knowledge.ProcessKnowledgeFromMultipleImages(
		ctx,
		g,
		"test-multi-image",
		imageIterator,
		logger,
		config.NewKnowledgeConfig(),
		embedder,
		nil,
	)

	require.NoError(t, err)
	require.NotNil(t, knowledgeResult)

	// Validate knowledge structure
	assert.Equal(t, "test-multi-image", knowledgeResult.ID)
	assert.Equal(t, "image", knowledgeResult.Metadata["source_type"])
	assert.Equal(t, 2, knowledgeResult.Metadata["image_count"])
	assert.Len(t, knowledgeResult.Documents, 2, "Should have processed 2 images")

	// Validate each document
	for i, doc := range knowledgeResult.Documents {
		t.Logf("Image %d: ID=%s", i, doc.ID)

		// Check content structure
		assert.Equal(t, knowledge.ContentTypeImage, doc.Content.Type())
		assert.NotEmpty(t, doc.Content.Image, "Image %d should have image data", i)
		assert.Equal(t, "image/jpeg", doc.Content.MIMEType, "Image %d should be JPEG", i)

		// Check metadata
		assert.NotNil(t, doc.Metadata["image_number"], "Image %d should have image_number", i)
		assert.NotNil(t, doc.Metadata["timestamp"], "Image %d should have timestamp from input", i)
		assert.NotNil(t, doc.Metadata["frame_number"], "Image %d should have frame_number from input", i)

		// Check embeddings
		assert.NotEmpty(t, doc.Embeddings, "Image %d should have embeddings", i)
		assert.Len(t, doc.Embeddings, 768, "Image %d should have 768-dimensional embedding", i)
	}

	t.Logf("Successfully processed %d images with embeddings", len(knowledgeResult.Documents))
}

// TestIndexKnowledgeFromImages tests the service-level image indexing
func TestIndexKnowledgeFromImages(t *testing.T) {
	ctx := t.Context()

	// Check if we have NOMIC API key
	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	// Initialize service
	modelConfig := &config.ModelConfig{}
	knowledgeConfig := config.NewKnowledgeConfig()
	knowledgeConfig.NomicAPIKey = nomicApiKey

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	service, err := knowledge.NewService(ctx, modelConfig, knowledgeConfig, logger)
	require.NoError(t, err)
	defer service.Close()

	// Create image iterator
	imageIterator := func(yield func(*knowledge.ImageReader, error) bool) {
		yield(&knowledge.ImageReader{
			Content:     bytes.NewReader(testImage1PNG),
			ContentType: "image/png",
			Metadata: map[string]any{
				"description": "First test image",
			},
		}, nil)
	}

	// Index knowledge from images
	result, err := service.IndexKnowledgeFromImages(ctx, "test-image-knowledge", imageIterator, nil)

	require.NoError(t, err)
	require.NotNil(t, result)

	// Validate result
	assert.Equal(t, "test-image-knowledge", result.ID)
	assert.Equal(t, "image", result.Metadata["source_type"])
	assert.Greater(t, len(result.Documents), 0, "Should have indexed images")

	// Test retrieval
	retrieved, err := service.GetKnowledge(ctx, "test-image-knowledge")
	require.NoError(t, err)
	assert.Equal(t, result.ID, retrieved.ID)

	t.Logf("Indexed image knowledge with %d images", len(result.Documents))
}

// TestProcessDocumentFromImage_InvalidInput tests error handling
func TestProcessDocumentFromImage_InvalidInput(t *testing.T) {
	ctx := t.Context()

	// Initialize genkit
	modelConfig := &config.ModelConfig{}
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	g := xgenkit.NewGenkit(ctx, modelConfig, logger, false)

	tests := []struct {
		name        string
		input       []byte
		contentType string
		expectedErr string
	}{
		{
			name:        "empty data",
			input:       []byte{},
			contentType: "image/jpeg",
			expectedErr: "empty image data",
		},
		{
			name:        "invalid image",
			input:       []byte("not an image"),
			contentType: "image/jpeg",
			expectedErr: "failed to decode image",
		},
		{
			name:        "corrupted image",
			input:       []byte("\xFF\xD8\xFF\xE0\x00\x10JFIF"),
			contentType: "image/jpeg",
			expectedErr: "failed to decode image",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			imageReader := &knowledge.ImageReader{
				Content:     bytes.NewReader(tt.input),
				ContentType: tt.contentType,
			}

			_, err := knowledge.ProcessDocumentFromImage(ctx, g, imageReader, logger, 1)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErr)
		})
	}
}

// TestImageFormatConversion tests that all supported formats are converted to JPEG
func TestImageFormatConversion(t *testing.T) {
	ctx := t.Context()

	// Initialize genkit
	modelConfig := &config.ModelConfig{}
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	g := xgenkit.NewGenkit(ctx, modelConfig, logger, false)

	// Test PNG to JPEG conversion
	imageReader := &knowledge.ImageReader{
		Content:     bytes.NewReader(testImage1PNG),
		ContentType: "image/png",
	}

	document, err := knowledge.ProcessDocumentFromImage(ctx, g, imageReader, logger, 1)
	require.NoError(t, err)

	// Should be converted to JPEG
	assert.Equal(t, "image/jpeg", document.Content.MIMEType)
	assert.Equal(t, "png", document.Metadata["original_format"])
	assert.NotEmpty(t, document.Content.Image)

	t.Logf("Successfully converted PNG to JPEG")
}

// TestImageWithCustomMetadata tests that custom metadata is preserved
func TestImageWithCustomMetadata(t *testing.T) {
	ctx := t.Context()

	// Check if we have NOMIC API key
	nomicApiKey := os.Getenv("NOMIC_API_KEY")
	if nomicApiKey == "" {
		t.Skip("NOMIC_API_KEY not set, skipping test")
	}

	// Initialize genkit
	modelConfig := &config.ModelConfig{}
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	g := xgenkit.NewGenkit(ctx, modelConfig, logger, false)

	customMetadata := map[string]any{
		"video_name":   "sample_video.mp4",
		"timestamp":    5.5,
		"frame_number": 165,
		"description":  "A frame showing a car",
	}

	imageIterator := func(yield func(*knowledge.ImageReader, error) bool) {
		yield(&knowledge.ImageReader{
			Content:     bytes.NewReader(testImage1PNG),
			ContentType: "image/png",
			Metadata:    customMetadata,
		}, nil)
	}

	embedder := knowledge.NewEmbedder(nomicApiKey)

	knowledgeResult, err := knowledge.ProcessKnowledgeFromMultipleImages(
		ctx,
		g,
		"test-metadata",
		imageIterator,
		logger,
		config.NewKnowledgeConfig(),
		embedder,
		nil,
	)

	require.NoError(t, err)
	require.Len(t, knowledgeResult.Documents, 1)

	doc := knowledgeResult.Documents[0]

	// Check that custom metadata is preserved
	assert.Equal(t, "sample_video.mp4", doc.Metadata["video_name"])
	assert.Equal(t, 5.5, doc.Metadata["timestamp"])
	assert.Equal(t, 165, doc.Metadata["frame_number"])
	assert.Equal(t, "A frame showing a car", doc.Metadata["description"])

	t.Logf("Custom metadata preserved: %+v", doc.Metadata)
}
