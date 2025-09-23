package knowledge

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"strings"

	"github.com/firebase/genkit/go/genkit"
	"github.com/habiliai/agentruntime/config"
	"github.com/pkg/errors"
)

const (
	SourceTypeDocument = "document"
	SourceTypeCSV      = "csv"
	SourceTypeJSON     = "json"
	SourceTypeText     = "text"
	SourceTypeMarkdown = "markdown"
)

// ProcessKnowledgeFromMultipleDocuments processes multiple documents with different types and merges them into a single Knowledge object
func ProcessKnowledgeFromMultipleDocuments(
	ctx context.Context,
	g *genkit.Genkit,
	id string,
	inputs iter.Seq2[*DocumentReader, error],
	logger *slog.Logger,
	config *config.KnowledgeConfig,
	embedder Embedder,
) (*Knowledge, error) {
	// Create knowledge object
	knowledge := &Knowledge{
		ID: id,
		Metadata: map[string]any{
			MetadataKeySourceType: SourceTypeDocument,
			"document_count":      0,
			"document_types":      make([]string, 0),
		},
		Documents: make([]*Document, 0),
	}

	documentCount := 0
	documentTypes := make(map[string]int) // Track document type counts
	globalDocumentIndex := 1

	// Process each document directly
	for docReader, err := range inputs {
		if err != nil {
			return nil, err
		}
		documentCount++

		// Process document based on its content type
		documents, docMetadata, err := ProcessDocumentsByType(ctx, g, docReader, logger, config, embedder, globalDocumentIndex)
		if err != nil {
			logger.Warn("Failed to process document",
				"document_number", documentCount,
				"content_type", docReader.ContentType,
				"error", err.Error())
			continue
		}

		// Track document type
		sourceType := getSourceTypeFromContentType(docReader.ContentType)
		documentTypes[sourceType]++

		// Add documents with updated IDs and metadata
		for _, doc := range documents {
			// Update document ID to include document number and global index
			doc.ID = fmt.Sprintf("%s_doc_%d_%d", id, documentCount, globalDocumentIndex)

			// Update metadata to include document source info
			if doc.Metadata == nil {
				doc.Metadata = make(map[string]any)
			}
			doc.Metadata["document_number"] = documentCount
			doc.Metadata["global_index"] = globalDocumentIndex
			doc.Metadata["source_content_type"] = docReader.ContentType
			doc.Metadata["source_type"] = sourceType

			// Merge document-level metadata
			for k, v := range docMetadata {
				if _, exists := doc.Metadata[k]; !exists {
					doc.Metadata[k] = v
				}
			}

			knowledge.Documents = append(knowledge.Documents, doc)
			globalDocumentIndex++
		}
	}

	if len(knowledge.Documents) == 0 {
		return nil, errors.Errorf("no valid documents found for knowledge %s", id)
	}

	// Update metadata with collected information
	knowledge.Metadata["document_count"] = documentCount
	knowledge.Metadata["total_chunks"] = globalDocumentIndex - 1

	// Convert document types map to array for metadata
	typesList := make([]string, 0, len(documentTypes))
	for docType, count := range documentTypes {
		typesList = append(typesList, fmt.Sprintf("%s:%d", docType, count))
	}
	knowledge.Metadata["document_types"] = typesList

	logger.Info("Processed multiple documents",
		"document_count", documentCount,
		"total_chunks", len(knowledge.Documents),
		"knowledge_id", id,
		"document_types", typesList)

	return knowledge, nil
}

// ProcessDocumentsByType processes a single document based on its content type
func ProcessDocumentsByType(
	ctx context.Context,
	g *genkit.Genkit,
	docReader *DocumentReader,
	logger *slog.Logger,
	config *config.KnowledgeConfig,
	embedder Embedder,
	startIndex int,
) ([]*Document, map[string]any, error) {
	switch docReader.ContentType {
	case "application/pdf":
		return ProcessDocumentsFromPDF(ctx, g, docReader.Content, logger, config, embedder)

	case "text/csv":
		return ProcessDocumentsFromCSV(ctx, docReader.Content, logger, embedder, startIndex)

	case "application/json", "text/json":
		return ProcessDocumentsFromJSON(ctx, docReader.Content, logger, embedder, startIndex)

	case "text/plain":
		return ProcessDocumentsFromText(ctx, docReader.Content, logger, embedder, startIndex)

	case "text/markdown":
		return ProcessDocumentsFromMarkdown(ctx, docReader.Content, logger, embedder, startIndex)

	default:
		// Try to process as plain text for unknown types
		logger.Warn("Unknown content type, processing as plain text", "content_type", docReader.ContentType)
		return ProcessDocumentsFromText(ctx, docReader.Content, logger, embedder, startIndex)
	}
}

// ProcessDocumentsFromCSV processes CSV content and returns documents
func ProcessDocumentsFromCSV(
	ctx context.Context,
	reader io.Reader,
	logger *slog.Logger,
	embedder Embedder,
	startIndex int,
) ([]*Document, map[string]any, error) {
	csvReader := csv.NewReader(reader)
	records, err := csvReader.ReadAll()
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to read CSV")
	}

	if len(records) == 0 {
		return nil, nil, errors.New("empty CSV file")
	}

	documents := make([]*Document, 0)
	var headers []string

	// First row as headers
	if len(records) > 0 {
		headers = records[0]
		records = records[1:] // Skip header row
	}

	metadata := map[string]any{
		"source_type": SourceTypeCSV,
		"total_rows":  len(records),
		"columns":     headers,
	}

	// Create a document for each CSV row
	for i, record := range records {
		if len(record) != len(headers) {
			logger.Warn("CSV row column count mismatch", "row", i+1, "expected", len(headers), "got", len(record))
			continue
		}

		// Create text representation of the row
		var textParts []string
		rowData := make(map[string]string)

		for j, value := range record {
			if j < len(headers) {
				header := headers[j]
				rowData[header] = value
				if value != "" {
					textParts = append(textParts, fmt.Sprintf("%s: %s", header, value))
				}
			}
		}

		if len(textParts) == 0 {
			continue // Skip empty rows
		}

		content := strings.Join(textParts, " | ")

		doc := &Document{
			Content: Content{
				Text:     content,
				MIMEType: "text/plain",
			},
			EmbeddingText: content,
			Metadata: map[string]any{
				"row_number": i + 1,
				"row_data":   rowData,
			},
		}

		documents = append(documents, doc)
	}

	if len(documents) == 0 {
		return nil, nil, errors.New("no valid rows found in CSV")
	}

	// Generate embeddings for all documents
	embeddingTexts := make([]string, len(documents))
	for i, doc := range documents {
		embeddingTexts[i] = doc.EmbeddingText
	}

	embeddings, err := embedder.EmbedTexts(ctx, EmbeddingTaskTypeDocument, embeddingTexts...)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to generate embeddings for CSV")
	}

	if len(embeddings) != len(documents) {
		return nil, nil, errors.Errorf("embedding count mismatch: got %d, expected %d", len(embeddings), len(documents))
	}

	// Assign embeddings to documents
	for i := range documents {
		documents[i].Embeddings = embeddings[i]
	}

	logger.Info("Processed CSV", "rows", len(documents), "columns", len(headers))
	return documents, metadata, nil
}

// ProcessDocumentsFromJSON processes JSON content and returns documents
func ProcessDocumentsFromJSON(
	ctx context.Context,
	reader io.Reader,
	logger *slog.Logger,
	embedder Embedder,
	startIndex int,
) ([]*Document, map[string]any, error) {
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to read JSON")
	}

	// Try to parse as array first
	var jsonArray []map[string]any
	if err := json.Unmarshal(data, &jsonArray); err == nil {
		// Process as array of objects
		return processJSONArray(ctx, jsonArray, logger, embedder)
	}

	// Try to parse as single object
	var jsonObject map[string]any
	if err := json.Unmarshal(data, &jsonObject); err == nil {
		// Process as single object
		return processJSONObject(ctx, jsonObject, logger, embedder)
	}

	return nil, nil, errors.New("failed to parse JSON as array or object")
}

func processJSONArray(
	ctx context.Context,
	jsonArray []map[string]any,
	logger *slog.Logger,
	embedder Embedder,
) ([]*Document, map[string]any, error) {
	documents := ProcessKnowledgeFromMap(jsonArray)

	metadata := map[string]any{
		"source_type": SourceTypeJSON,
		"total_items": len(jsonArray),
		"data_type":   "array",
	}

	if len(documents) == 0 {
		return nil, nil, errors.New("no valid items found in JSON array")
	}

	// Generate embeddings for all documents
	embeddingTexts := make([]string, len(documents))
	for i, doc := range documents {
		embeddingTexts[i] = doc.EmbeddingText
	}

	embeddings, err := embedder.EmbedTexts(ctx, EmbeddingTaskTypeDocument, embeddingTexts...)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to generate embeddings for JSON array")
	}

	if len(embeddings) != len(documents) {
		return nil, nil, errors.Errorf("embedding count mismatch: got %d, expected %d", len(embeddings), len(documents))
	}

	// Assign embeddings to documents
	for i := range documents {
		documents[i].Embeddings = embeddings[i]
	}

	logger.Info("Processed JSON array", "items", len(documents))
	return documents, metadata, nil
}

func processJSONObject(
	ctx context.Context,
	jsonObject map[string]any,
	logger *slog.Logger,
	embedder Embedder,
) ([]*Document, map[string]any, error) {
	documents := ProcessKnowledgeFromMap([]map[string]any{jsonObject})

	metadata := map[string]any{
		"source_type": SourceTypeJSON,
		"total_items": 1,
		"data_type":   "object",
	}

	if len(documents) == 0 {
		return nil, nil, errors.New("no valid content found in JSON object")
	}

	// Generate embeddings for the document
	embeddings, err := embedder.EmbedTexts(ctx, EmbeddingTaskTypeDocument, documents[0].EmbeddingText)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to generate embeddings for JSON object")
	}

	if len(embeddings) != 1 {
		return nil, nil, errors.Errorf("embedding count mismatch: got %d, expected 1", len(embeddings))
	}

	documents[0].Embeddings = embeddings[0]

	logger.Info("Processed JSON object", "keys", len(jsonObject))
	return documents, metadata, nil
}

// ProcessDocumentsFromText processes plain text content
func ProcessDocumentsFromText(
	ctx context.Context,
	reader io.Reader,
	logger *slog.Logger,
	embedder Embedder,
	startIndex int,
) ([]*Document, map[string]any, error) {
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to read text")
	}

	text := strings.TrimSpace(string(data))
	if text == "" {
		return nil, nil, errors.New("empty text content")
	}

	// Split text into chunks (simple split by paragraphs)
	chunks := splitTextIntoChunks(text, 1000) // 1000 character chunks

	documents := make([]*Document, 0, len(chunks))
	for i, chunk := range chunks {
		if strings.TrimSpace(chunk) == "" {
			continue
		}

		doc := &Document{
			Content: Content{
				Text:     chunk,
				MIMEType: "text/plain",
			},
			EmbeddingText: chunk,
			Metadata: map[string]any{
				"chunk_number": i + 1,
			},
		}
		documents = append(documents, doc)
	}

	metadata := map[string]any{
		"source_type": SourceTypeText,
		"total_chars": len(text),
		"chunk_count": len(documents),
	}

	if len(documents) == 0 {
		return nil, nil, errors.New("no valid content found in text")
	}

	// Generate embeddings
	embeddingTexts := make([]string, len(documents))
	for i, doc := range documents {
		embeddingTexts[i] = doc.EmbeddingText
	}

	embeddings, err := embedder.EmbedTexts(ctx, EmbeddingTaskTypeDocument, embeddingTexts...)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to generate embeddings for text")
	}

	if len(embeddings) != len(documents) {
		return nil, nil, errors.Errorf("embedding count mismatch: got %d, expected %d", len(embeddings), len(documents))
	}

	// Assign embeddings
	for i := range documents {
		documents[i].Embeddings = embeddings[i]
	}

	logger.Info("Processed text", "chunks", len(documents), "chars", len(text))
	return documents, metadata, nil
}

// ProcessDocumentsFromMarkdown processes markdown content (similar to text but with markdown awareness)
func ProcessDocumentsFromMarkdown(
	ctx context.Context,
	reader io.Reader,
	logger *slog.Logger,
	embedder Embedder,
	startIndex int,
) ([]*Document, map[string]any, error) {
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to read markdown")
	}

	text := strings.TrimSpace(string(data))
	if text == "" {
		return nil, nil, errors.New("empty markdown content")
	}

	// Split markdown by headers and sections
	chunks := splitMarkdownIntoChunks(text)

	documents := make([]*Document, 0, len(chunks))
	for i, chunk := range chunks {
		if strings.TrimSpace(chunk) == "" {
			continue
		}

		doc := &Document{
			Content: Content{
				Text:     chunk,
				MIMEType: "text/markdown",
			},
			EmbeddingText: chunk,
			Metadata: map[string]any{
				"section_number": i + 1,
			},
		}
		documents = append(documents, doc)
	}

	metadata := map[string]any{
		"source_type":   SourceTypeMarkdown,
		"total_chars":   len(text),
		"section_count": len(documents),
	}

	if len(documents) == 0 {
		return nil, nil, errors.New("no valid content found in markdown")
	}

	// Generate embeddings
	embeddingTexts := make([]string, len(documents))
	for i, doc := range documents {
		embeddingTexts[i] = doc.EmbeddingText
	}

	embeddings, err := embedder.EmbedTexts(ctx, EmbeddingTaskTypeDocument, embeddingTexts...)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to generate embeddings for markdown")
	}

	if len(embeddings) != len(documents) {
		return nil, nil, errors.Errorf("embedding count mismatch: got %d, expected %d", len(embeddings), len(documents))
	}

	// Assign embeddings
	for i := range documents {
		documents[i].Embeddings = embeddings[i]
	}

	logger.Info("Processed markdown", "sections", len(documents), "chars", len(text))
	return documents, metadata, nil
}

// Helper functions

func getSourceTypeFromContentType(contentType string) string {
	switch contentType {
	case "application/pdf":
		return SourceTypePDF
	case "text/csv":
		return SourceTypeCSV
	case "application/json", "text/json":
		return SourceTypeJSON
	case "text/markdown":
		return SourceTypeMarkdown
	case "text/plain":
		return SourceTypeText
	default:
		return SourceTypeText
	}
}

func splitTextIntoChunks(text string, maxChunkSize int) []string {
	if len(text) <= maxChunkSize {
		return []string{text}
	}

	chunks := make([]string, 0)
	paragraphs := strings.Split(text, "\n\n")

	currentChunk := ""
	for _, paragraph := range paragraphs {
		if len(currentChunk)+len(paragraph)+2 <= maxChunkSize {
			if currentChunk != "" {
				currentChunk += "\n\n"
			}
			currentChunk += paragraph
		} else {
			if currentChunk != "" {
				chunks = append(chunks, currentChunk)
			}
			// If paragraph is too long, split it by sentences
			if len(paragraph) > maxChunkSize {
				sentences := strings.Split(paragraph, ". ")
				currentChunk = ""
				for _, sentence := range sentences {
					if len(currentChunk)+len(sentence)+2 <= maxChunkSize {
						if currentChunk != "" {
							currentChunk += ". "
						}
						currentChunk += sentence
					} else {
						if currentChunk != "" {
							chunks = append(chunks, currentChunk)
						}
						currentChunk = sentence
					}
				}
			} else {
				currentChunk = paragraph
			}
		}
	}

	if currentChunk != "" {
		chunks = append(chunks, currentChunk)
	}

	return chunks
}

func splitMarkdownIntoChunks(text string) []string {
	lines := strings.Split(text, "\n")
	chunks := make([]string, 0)

	currentChunk := ""
	currentLevel := 0

	for _, line := range lines {
		// Check if this is a header
		if strings.HasPrefix(line, "#") {
			headerLevel := 0
			for _, char := range line {
				if char == '#' {
					headerLevel++
				} else {
					break
				}
			}

			// If we have a new section at same or higher level, start new chunk
			if headerLevel <= currentLevel && currentChunk != "" {
				chunks = append(chunks, strings.TrimSpace(currentChunk))
				currentChunk = ""
			}
			currentLevel = headerLevel
		}

		if currentChunk != "" {
			currentChunk += "\n"
		}
		currentChunk += line
	}

	if currentChunk != "" {
		chunks = append(chunks, strings.TrimSpace(currentChunk))
	}

	return chunks
}
