package tool

import (
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/habiliai/agentruntime/entity"
)

type Knowledge struct {
	ai.Media `json:",inline"`
	Score    float64 `json:"score,omitempty" jsonschema:"description=Score of the search result"`
	Context  string  `json:"context,omitempty" jsonschema:"description=Text of the search result"`
}

func (m *manager) registerKnowledgeSearchTool(skill *entity.NativeAgentSkill) error {
	allowedKnowledgeIds, ok := skill.Env["knowledge_ids"].([]string)
	if !ok {
		allowedKnowledgeIds = nil
	}

	return registerNativeTool(
		m,
		"knowledge_search",
		`Search through the external knowledge base to find relevant information, documents, and context.

This tool provides semantic search capabilities across stored knowledge including:
- Previously saved documents, notes, and information
- Context from past conversations and interactions
- Domain-specific knowledge and reference materials
- User preferences, patterns, and historical data

When to use this tool:
- When asked about information that might be stored in the knowledge base
- To retrieve context from previous conversations or saved documents
- When you need background information before answering a question
- To verify facts or find supporting evidence from stored knowledge
- When the user references something from the past ("as we discussed", "like last time", etc.)

How to use:
- Provide a clear, specific search query as input
- Use keywords and phrases that describe what you're looking for
- Consider multiple search attempts with different phrasings if initial results are insufficient
- The tool returns relevant excerpts with metadata about the source

Response structure:
- Results: Array of matching knowledge items with content and metadata
- Count: Number of results returned (may be less than limit if fewer matches found)
- Error: Error message if the search fails (e.g., connection issues, invalid query)

Error handling:
- If an error occurs, the Error field will contain the error message
- Common errors: database connection issues, invalid query format, service unavailable
- When an error occurs, try rephrasing the query or waiting before retry
- The tool will still return a response (not throw an error) to allow graceful handling

The search uses semantic similarity, so exact keyword matches are not required. Results are ranked by relevance and include context about when and where the information was stored.`,
		skill,
		func(ctx *Context, input struct {
			Query string `json:"query" jsonschema:"description=The search query to find relevant information"`
			Limit *int   `json:"limit,omitempty" jsonschema:"description=The maximum number of results to return,default=5"`
		}) (reply struct {
			Output []Knowledge `json:"output,omitempty" jsonschema:"description=List of search results with relevant knowledge"`
			Error  string      `json:"error,omitempty" jsonschema:"description=Error message if the search fails"`
		}, err error) {
			// Set default limit if not provided
			limit := 5
			if input.Limit != nil {
				limit = *input.Limit
			}

			// Retrieve relevant knowledge
			results, err := m.knowledgeService.RetrieveRelevantKnowledge(ctx, input.Query, limit, allowedKnowledgeIds)
			if err != nil {
				reply.Error = err.Error()
				return reply, nil
			}

			// Clean up embedding data to reduce response size
			for _, res := range results {
				k := Knowledge{
					Score: float64(res.Score),
				}
				switch res.Content.MIMEType {
				case "image/jpeg", "image/png", "image/jpg", "image/webp":
					k.Media = ai.Media{
						ContentType: res.Content.MIMEType,
						Url:         res.Content.Image,
					}
				case "text/plain", "plain/text", "text/markdown":
					k.Context = res.Content.Text
				default:
					return reply, fmt.Errorf("unknown content type: %s", res.Content.MIMEType)
				}
				reply.Output = append(reply.Output, k)
			}

			return
		},
	)
}
