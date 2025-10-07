package knowledge

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"iter"
	"log/slog"
	"time"

	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/config"
	"github.com/pkg/errors"
	"golang.org/x/image/webp"
)

const (
	SourceTypeImage = "image"
)

// IndexKnowledgeFromImages processes image files and creates searchable knowledge from images
func (s *service) IndexKnowledgeFromImages(ctx context.Context, id string, input iter.Seq2[*ImageReader, error], metadata map[string]any) (*Knowledge, error) {
	// First, delete existing knowledge for this ID
	if id != "" {
		if err := s.DeleteKnowledge(ctx, id); err != nil {
			return nil, errors.Wrapf(err, "failed to delete existing knowledge")
		}
	}

	knowledge, err := ProcessKnowledgeFromMultipleImages(ctx, s.genkit, id, input, s.logger, s.config, s.embedder, metadata)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to process knowledge from images")
	}

	// Store all items
	if err := s.store.Store(ctx, knowledge); err != nil {
		return nil, errors.Wrapf(err, "failed to store knowledge")
	}

	return knowledge, nil
}

// ProcessKnowledgeFromMultipleImages processes multiple image readers and merges them into a single Knowledge object
func ProcessKnowledgeFromMultipleImages(
	ctx context.Context,
	g *genkit.Genkit,
	id string,
	inputs iter.Seq2[*ImageReader, error],
	logger *slog.Logger,
	config *config.KnowledgeConfig,
	embedder Embedder,
	metadata map[string]any,
) (*Knowledge, error) {
	// Create knowledge object
	knowledge := &Knowledge{
		ID: id,
		Metadata: map[string]any{
			MetadataKeySourceType: SourceTypeImage,
			"image_count":         0,
		},
		Documents: make([]*Document, 0),
	}

	imageCount := 0
	globalImageNumber := 1

	// Process each image directly
	for input, err := range inputs {
		if err != nil {
			return nil, err
		}
		imageCount++

		// Process image directly to get document
		document, err := ProcessDocumentFromImage(ctx, g, input, logger, globalImageNumber)
		if err != nil {
			logger.Warn("Failed to process image", "image_number", imageCount, "error", err.Error())
			continue
		}

		// Update document ID to include image number
		document.ID = fmt.Sprintf("%s_image_%d", id, globalImageNumber)

		// Update metadata to include image source info
		if document.Metadata == nil {
			document.Metadata = make(map[string]any)
		}
		document.Metadata["image_number"] = imageCount
		document.Metadata["global_image_number"] = globalImageNumber

		// Merge user-provided metadata from ImageReader
		for k, v := range input.Metadata {
			document.Metadata[k] = v
		}

		// Merge user-provided metadata from ImageReader
		for k, v := range metadata {
			document.Metadata[k] = v
		}

		knowledge.Documents = append(knowledge.Documents, document)
		globalImageNumber++
	}

	if len(knowledge.Documents) == 0 {
		return nil, errors.Errorf("no valid images found for knowledge %s", id)
	}

	// Update metadata with collected information
	knowledge.Metadata["image_count"] = imageCount
	knowledge.Metadata["total_images"] = globalImageNumber - 1

	logger.Info("Processed multiple images",
		"image_count", imageCount,
		"total_images", len(knowledge.Documents),
		"knowledge_id", id)

	// Generate embeddings for all images using vision embedder
	now := time.Now()
	imageData := make([][]byte, len(knowledge.Documents))
	for i, doc := range knowledge.Documents {
		// Decode base64 to raw bytes
		imgBytes, err := base64.StdEncoding.DecodeString(doc.Content.Image)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to decode image %d", i)
		}
		imageData[i] = imgBytes
	}

	embeddings, err := embedder.EmbedImageFiles(ctx, "image/jpeg", imageData...)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to generate vision embeddings")
	}

	if len(embeddings) != len(knowledge.Documents) {
		return nil, errors.Errorf("embedding count mismatch: got %d, expected %d", len(embeddings), len(knowledge.Documents))
	}

	// Assign embeddings to documents
	for i := range knowledge.Documents {
		knowledge.Documents[i].Embeddings = embeddings[i]
	}

	logger.Info("Generated vision embeddings", "embedding_count", len(embeddings), "time", time.Since(now))

	for k, v := range metadata {
		knowledge.Metadata[k] = v
	}

	return knowledge, nil
}

// ProcessDocumentFromImage processes a single image and returns a document
func ProcessDocumentFromImage(
	ctx context.Context,
	g *genkit.Genkit,
	input *ImageReader,
	logger *slog.Logger,
	imageNumber int,
) (*Document, error) {
	// Read image data into memory
	imageData, err := io.ReadAll(input.Content)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read image data")
	}

	if len(imageData) == 0 {
		return nil, errors.New("empty image data")
	}

	// Decode image to get dimensions
	img, format, err := decodeImage(bytes.NewReader(imageData), input.ContentType)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to decode image")
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	logger.Debug("Decoded image",
		"number", imageNumber,
		"format", format,
		"width", width,
		"height", height,
		"size_bytes", len(imageData))

	// Convert to JPEG if needed for consistent storage
	var jpegData []byte
	if format == "jpeg" {
		jpegData = imageData
	} else {
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85}); err != nil {
			return nil, errors.Wrap(err, "failed to encode image as JPEG")
		}
		jpegData = buf.Bytes()
	}

	// Convert to base64
	base64Image := base64.StdEncoding.EncodeToString(jpegData)

	// Create document for this image
	document := &Document{
		Content: Content{
			Image:    base64Image,
			MIMEType: "image/jpeg",
		},
		EmbeddingText: fmt.Sprintf("Image %d", imageNumber),
		Metadata: map[string]any{
			"image_number":    imageNumber,
			"original_format": format,
			"width":           width,
			"height":          height,
			"size_bytes":      len(imageData),
		},
	}

	return document, nil
}

// decodeImage decodes an image from reader based on content type
func decodeImage(r io.Reader, contentType string) (image.Image, string, error) {
	switch contentType {
	case "image/jpeg", "image/jpg":
		img, err := jpeg.Decode(r)
		return img, "jpeg", err
	case "image/png":
		img, err := png.Decode(r)
		return img, "png", err
	case "image/gif":
		img, err := gif.Decode(r)
		return img, "gif", err
	case "image/webp":
		img, err := webp.Decode(r)
		return img, "webp", err
	default:
		// Try to auto-detect
		img, format, err := image.Decode(r)
		if err != nil {
			return nil, "", errors.Wrapf(err, "unsupported image type: %s", contentType)
		}
		return img, format, nil
	}
}
