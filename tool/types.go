package tool

import (
	"regexp"
	"time"
)

// MCPTransportType represents the transport type for MCP servers
type MCPTransportType string

const (
	MCPTransportStdio    MCPTransportType = "stdio"
	MCPTransportSSE      MCPTransportType = "sse"
	MCPTransportOAuthSSE MCPTransportType = "oauth-sse"
	MCPTransportHTTP     MCPTransportType = "http"
)

// MCPServerConfig represents the configuration for an MCP server
type MCPServerConfig struct {
	// Transport type (stdio, sse, oauth-sse, http)
	Transport MCPTransportType `json:"transport,omitempty" yaml:"transport,omitempty"`

	// For stdio transport
	Command string         `json:"command,omitempty" yaml:"command,omitempty"`
	Args    []string       `json:"args,omitempty" yaml:"args,omitempty"`
	Env     map[string]any `json:"env,omitempty" yaml:"env,omitempty"`

	// For SSE/HTTP transports
	URL     string            `json:"url,omitempty" yaml:"url,omitempty"`
	Headers map[string]string `json:"headers,omitempty" yaml:"headers,omitempty"`

	// For OAuth
	OAuthConfig *OAuthConfig `json:"oauth,omitempty" yaml:"oauth,omitempty"`

	// Connection settings
	Timeout           time.Duration `json:"timeout,omitempty" yaml:"timeout,omitempty"`
	KeepAliveInterval time.Duration `json:"keepAliveInterval,omitempty" yaml:"keepAliveInterval,omitempty"`
}

// OAuthConfig contains OAuth authentication configuration
type OAuthConfig struct {
	// ClientID is the OAuth client ID
	ClientID string `json:"clientId,omitempty"`
	// ClientSecret is the OAuth client secret (for confidential clients)
	ClientSecret string `json:"clientSecret,omitempty"`
	// RedirectURL is the redirect URI for the OAuth flow
	RedirectURL string `json:"redirectUrl,omitempty"`
	// Scopes is the list of OAuth scopes to request
	Scopes []string `json:"scopes,omitempty"`
	// AuthServerMetadataURL is the URL to the OAuth server metadata
	// If empty, the client will attempt to discover it from the base URL
	AuthServerMetadataURL string `json:"authServerMetadataUrl,omitempty"`
	// PKCEEnabled enables PKCE for the OAuth flow (recommended for public clients)
	PKCEEnabled bool `json:"pkceEnabled,omitempty"`
}

// GetTransport returns the transport type, defaulting to stdio if not specified
func (c *MCPServerConfig) GetTransport() MCPTransportType {
	if c.Transport == "" {
		// Auto-detect based on config
		if c.URL != "" {
			return MCPTransportSSE
		}
		return MCPTransportStdio
	}
	return c.Transport
}

func CanBeUsedAsToolName(toolName string) bool {
	// regexp: ([a-zA-Z0-9_-]{1,128})
	re := regexp.MustCompile(`^[a-zA-Z0-9_-]{1,128}$`)
	return re.MatchString(toolName)
}
