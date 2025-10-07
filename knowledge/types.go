package knowledge

import (
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/mokiat/gog"
)

type (
	Knowledge struct {
		ID        string         `json:"id"`
		Metadata  map[string]any `json:"metadata"`
		Documents []*Document    `json:"documents"`
	}

	Document struct {
		ID            string         `json:"id"`
		Content       Content        `json:"content"`
		Embeddings    []float32      `json:"embeddings"`
		EmbeddingText string         `json:"embeddingText"`
		Metadata      map[string]any `json:"metadata"`
	}

	KnowledgeSearchResult struct {
		*Document `json:",inline"`
		Score     float32 `json:"score"`
	}

	Content struct {
		Text     string `json:"text,omitempty"`
		Image    string `json:"data,omitempty"`
		MIMEType string `json:"mimeType,omitempty"`
	}

	ContentType = string
)

const (
	// Source type constants for metadata
	SourceTypeMap   = "map"
	SourceTypePDF   = "pdf"
	SourceTypeVideo = "video"

	// Metadata keys for source information
	MetadataKeySourceTitle    = "source_title"
	MetadataKeySourceURL      = "source_url"
	MetadataKeySourceFilename = "source_filename"
	MetadataKeySourceType     = "source_type"

	ContentTypeText  ContentType = "text"
	ContentTypeImage ContentType = "image"
)

func (d *Document) ToDoc() (*ai.Document, error) {
	doc := &ai.Document{
		Metadata: gog.Merge(d.Metadata, map[string]any{
			"id": d.ID,
		}),
	}

	switch d.Content.MIMEType {
	case "text/plain", "plain/text", "text/markdown":
		doc.Content = append(doc.Content, ai.NewTextPart(d.Content.Text))
	case "image/jpeg", "image/jpg", "image/png", "image/webp":
		doc.Content = append(doc.Content, ai.NewMediaPart(d.Content.MIMEType, d.Content.Image))
	default:
		return nil, fmt.Errorf("unknown content type: %s", d.Content.MIMEType)
	}

	return doc, nil
}

func (c *Content) Type() ContentType {
	switch c.MIMEType {
	case "plain/text", "text/plain", "text/markdown":
		return ContentTypeText
	case "image/jpeg", "image/jpg", "image/png", "image/webp":
		return ContentTypeImage
	}
	return ""
}
