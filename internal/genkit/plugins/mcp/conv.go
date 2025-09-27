package mcp

import (
	"encoding/json"

	"github.com/mark3labs/mcp-go/mcp"
)

func makeInputSchema(
	schema mcp.ToolInputSchema,
) (map[string]any, error) {
	var inputSchema map[string]any

	schemaJson, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}

	if err = json.Unmarshal(schemaJson, &inputSchema); err != nil {
		return nil, err
	}

	return inputSchema, nil
}
