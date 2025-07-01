package knowledge

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"log/slog"
	"math"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/gen2brain/go-fitz"
	"github.com/habiliai/agentruntime/config"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mokiat/gog"
	"github.com/pkg/errors"
)

func (s *service) IndexKnowledgeFromPDF(ctx context.Context, id string, input io.Reader) (*Knowledge, error) {
	if s.embedder == nil {
		return nil, errors.New("embedder is not available - check OpenAI API key configuration. Knowledge indexing requires a valid OpenAI API key")
	}

	// First, delete existing knowledge for this agent
	if id != "" {
		if err := s.DeleteKnowledge(ctx, id); err != nil {
			return nil, errors.Wrapf(err, "failed to delete existing knowledge")
		}
	}

	knowledge, err := ProcessKnowledgeFromPDF(ctx, s.genkit, id, input, s.logger, s.config)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to process knowledge from PDF")
	}

	now := time.Now()

	// Generate embeddings for all documents
	embeddings, err := s.embedder.Embed(ctx, gog.Map(knowledge.Documents, func(d *Document) string {
		return d.EmbeddingText
	})...)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to generate embeddings")
	}

	if len(embeddings) != len(knowledge.Documents) {
		return nil, errors.Errorf("embedding count mismatch: got %d, expected %d", len(embeddings), len(knowledge.Documents))
	}

	// Assign embeddings to documents
	for i := range knowledge.Documents {
		knowledge.Documents[i].Embeddings = embeddings[i]
	}

	s.logger.Info("Generated embeddings", "time", time.Since(now))

	// Store all items
	if err := s.store.Store(ctx, knowledge); err != nil {
		return nil, errors.Wrapf(err, "failed to store knowledge")
	}

	return knowledge, nil
}

func ProcessKnowledgeFromPDF(ctx context.Context, g *genkit.Genkit, id string, input io.Reader, logger *slog.Logger, config *config.KnowledgeConfig) (*Knowledge, error) {
	// Read PDF data
	pdfData, err := io.ReadAll(input)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read PDF data")
	}

	// Open PDF with go-fitz
	doc, err := fitz.NewFromMemory(pdfData)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open PDF")
	}
	defer doc.Close()

	// Get PDF metadata
	pdfMetadata := doc.Metadata()

	// Create knowledge object
	knowledge := &Knowledge{
		ID: id,
		Source: Source{
			Title: pdfMetadata["title"],
			Type:  SourceTypePDF,
		},
		Metadata: map[string]any{
			"author":   pdfMetadata["author"],
			"subject":  pdfMetadata["subject"],
			"keywords": pdfMetadata["keywords"],
			"creator":  pdfMetadata["creator"],
			"producer": pdfMetadata["producer"],
		},
		Documents: make([]*Document, 0),
	}

	now := time.Now()

	// Process each page
	pageCount := doc.NumPage()
	for pageNum := 0; pageNum < pageCount; pageNum++ {
		// Render page as image with lower DPI to reduce size
		// Use 120 DPI for balance between quality and size
		img, err := doc.ImageDPI(pageNum, 120)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to render page %d as image", pageNum+1)
		}

		// Resize image if it's too large to prevent token limit issues
		bounds := img.Bounds()
		maxWidth := 512
		maxHeight := 512

		// Calculate scaling factor if needed
		width := bounds.Dx()
		height := bounds.Dy()
		scale := 1.0

		if width > maxWidth || height > maxHeight {
			scaleW := float64(maxWidth) / float64(width)
			scaleH := float64(maxHeight) / float64(height)
			scale = math.Min(scaleW, scaleH)

			// Create a new resized image
			newWidth := int(float64(width) * scale)
			newHeight := int(float64(height) * scale)
			resized := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))

			// Simple nearest-neighbor resizing
			for y := 0; y < newHeight; y++ {
				for x := 0; x < newWidth; x++ {
					srcX := int(float64(x) / scale)
					srcY := int(float64(y) / scale)
					resized.Set(x, y, img.At(srcX, srcY))
				}
			}
			img = resized
		}

		// Convert image to JPEG and encode as base64 (JPEG is much smaller than PNG)
		var buf bytes.Buffer
		// Use JPEG with quality 85 for good balance of quality and size
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85}); err != nil {
			return nil, errors.Wrapf(err, "failed to convert page %d to JPEG", pageNum+1)
		}
		base64Image := base64.StdEncoding.EncodeToString(buf.Bytes())

		var extractedText string
		switch config.PDFExtractionMethod {
		case "llm":
			// Extract text using Vision LLM
			extractedText, err = ExtractTextWithVisionLLM(ctx, g, base64Image, pageNum+1)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to extract text from page %d", pageNum+1)
			}
		case "library":
			extractedText, err = doc.Text(pageNum)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to extract text from page %d", pageNum+1)
			}
		default:
			return nil, errors.Errorf("invalid PDF extraction method: %s", config.PDFExtractionMethod)
		}

		// Create document for this page
		document := &Document{
			ID: fmt.Sprintf("%s_page_%d", id, pageNum+1),
			Contents: []mcp.Content{
				mcp.TextContent{
					Type: "text",
					Text: extractedText,
				},
				mcp.ImageContent{
					Type:     "image",
					Data:     base64Image,
					MIMEType: "image/jpeg", // Changed from image/png
				},
			},
			EmbeddingText: extractedText,
			Metadata: map[string]any{
				"page_number":       pageNum + 1,
				"total_pages":       pageCount,
				"extraction_method": config.PDFExtractionMethod,
			},
		}

		knowledge.Documents = append(knowledge.Documents, document)
	}

	logger.Info("Extracted pages", "time", time.Since(now), "pages", len(knowledge.Documents))

	if len(knowledge.Documents) == 0 {
		return nil, errors.Errorf("no pages found in PDF for knowledge %s", id)
	}

	return knowledge, nil
}

// ExtractTextWithVisionLLM uses Vision LLM to extract text from an image
func ExtractTextWithVisionLLM(ctx context.Context, g *genkit.Genkit, base64Image string, pageNum int) (string, error) {
	// Use a vision-capable model (GPT-4o or Claude 4 Sonnet)
	model := genkit.LookupModel(g, "anthropic", "claude-4-sonnet")
	if model == nil {
		model = genkit.LookupModel(g, "openai", "gpt-4o-mini")
	}
	if model == nil {
		return "", errors.New("No vision-capable model available")
	}

	// Create the message with image for vision model
	message := &ai.Message{
		Role: ai.RoleUser,
		Content: []*ai.Part{
			ai.NewTextPart("Please extract all text content from this PDF page image. Include all visible text, preserving the structure as much as possible. If there are tables, try to represent them clearly. If there are charts or diagrams, describe their content briefly. Focus on extracting readable text content."),
			ai.NewMediaPart("image/jpeg", base64Image),
		},
	}

	// Call the vision model using genkit.Generate
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithMessages(message),
		ai.WithOutputFormat(ai.OutputFormatText),
	)
	if err != nil {
		return "", errors.Wrapf(err, "failed to extract text using Vision LLM for page %d", pageNum)
	}

	if resp == nil {
		return "", errors.New("empty response from Vision LLM")
	}

	// Extract text from response
	extractedText := resp.Text()

	return strings.TrimSpace(extractedText), nil
}
