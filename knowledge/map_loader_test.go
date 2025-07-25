package knowledge_test

import (
	"testing"

	"github.com/habiliai/agentruntime/knowledge"
	"github.com/stretchr/testify/require"
)

// Helper function to get text content from Document
func getDocumentText(doc *knowledge.Document) string {
	if doc != nil && doc.Content.Type() == knowledge.ContentTypeText {
		return doc.Content.Text
	}
	return ""
}

func TestKnowledgeProcessing(t *testing.T) {
	knowledgeData := []map[string]any{
		{
			"cityName": "Seoul",
			"aliases":  "Seoul, SEOUL, KOR, Korea",
			"info":     "Capital city of South Korea, known for technology and K-pop culture",
			"weather":  "Four distinct seasons with hot summers and cold winters",
		},
		{
			"cityName": "Tokyo",
			"aliases":  "Tokyo, TYO, Japan",
			"info":     "Capital city of Japan, largest metropolitan area in the world",
			"weather":  "Humid subtropical climate with hot, humid summers",
		},
	}

	documents := knowledge.ProcessKnowledgeFromMap(knowledgeData)
	require.Len(t, documents, 2)

	// Check that content is extracted properly
	seoulText := getDocumentText(documents[0])
	tokyoText := getDocumentText(documents[1])

	require.Contains(t, seoulText, "Seoul")
	require.Contains(t, seoulText, "South Korea")
	require.Contains(t, tokyoText, "Tokyo")
	require.Contains(t, tokyoText, "Japan")

	// Check that metadata is preserved
	require.Equal(t, knowledgeData[0], documents[0].Metadata)
	require.Equal(t, knowledgeData[1], documents[1].Metadata)

	// Check embedding text is set
	require.NotEmpty(t, documents[0].EmbeddingText)
	require.NotEmpty(t, documents[1].EmbeddingText)
}

func TestTextExtraction(t *testing.T) {
	testCases := []struct {
		name     string
		input    map[string]any
		expected string
	}{
		{
			name: "standard text fields",
			input: map[string]any{
				"title":       "Test Title",
				"description": "Test Description",
				"content":     "Test Content",
			},
			expected: "Test Content Test Description Test Title",
		},
		{
			name: "custom fields",
			input: map[string]any{
				"cityName": "Seoul",
				"country":  "South Korea",
				"info":     "Technology hub",
			},
			expected: "cityName: Seoul country: South Korea info: Technology hub",
		},
		{
			name: "mixed types",
			input: map[string]any{
				"name":        "Test",
				"count":       42,
				"active":      true,
				"description": "Valid text",
			},
			expected: "Valid text Test",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := knowledge.ExtractTextFromMap(tc.input)
			require.Equal(t, tc.expected, result)
		})
	}
}
