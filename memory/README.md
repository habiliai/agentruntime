# Memory Service with RAG Support

This package provides memory management and RAG (Retrieval-Augmented Generation) functionality for AgentRuntime. The implementation uses GORM entities with JSONB fields for storing knowledge data and embeddings.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AgentConfig   │───▶│ Memory Service  │───▶│   SQLite DB     │
│   Knowledge     │    │ (RAG enabled)   │    │ (GORM entities) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ OpenAI Embedder │
                    │ (text-embedding │
                    │  -3-small)      │
                    └─────────────────┘
```

## Key Components

### 1. Knowledge Entity (`entity/knowledge.go`)
```go
type Knowledge struct {
    gorm.Model
    AgentName string                              // Agent identifier
    Content   string                              // Searchable text content
    Metadata  datatypes.JSONType[map[string]any] // Original knowledge data
    Embedding datatypes.JSONType[[]float32]      // Vector embedding
}
```

### 2. Memory Service Interface
```go
type Service interface {
    SetContext(ctx context.Context, context *AgentContext) error
    GetContext(ctx context.Context, name string) (*AgentContext, error)
    
    // RAG functionality
    IndexKnowledge(ctx context.Context, agentName string, knowledge []map[string]any) error
    RetrieveRelevantKnowledge(ctx context.Context, agentName string, query string, limit int) ([]string, error)
    DeleteAgentKnowledge(ctx context.Context, agentName string) error
}
```

### 3. Embedder Interface
```go
type Embedder interface {
    Embed(ctx context.Context, texts ...string) ([][]float32, error)
}
```

## Features

### 📝 Text Processing
- **Smart Text Extraction**: Automatically extracts searchable text from knowledge maps
- **Standard Field Priority**: Looks for common fields like `text`, `content`, `description`, etc.
- **Fallback Extraction**: Extracts all string values when no standard fields found
- **Deterministic Output**: Sorted keys ensure consistent text extraction

### 🔍 Vector Search
- **OpenAI Embeddings**: Uses `text-embedding-3-small` model
- **Cosine Similarity**: In-memory similarity calculation for retrieval
- **Similarity Ranking**: Results sorted by relevance score

### 🛡️ Graceful Degradation
- **Embedder Fallback**: Functions return gracefully when embedder unavailable
- **Service Resilience**: Engine continues working without memory service
- **Error Handling**: Comprehensive error wrapping with context

## Usage

### 1. Agent Configuration
Simply add knowledge to your agent YAML:

```yaml
name: TravelAgent
model: openai/gpt-4o
knowledge:
  - cityName: "Seoul"
    aliases: "Seoul, SEOUL, KOR, Korea"
    info: "Capital city of South Korea"
    weather: "Four distinct seasons"
  - cityName: "Tokyo"
    aliases: "Tokyo, TYO, Japan" 
    info: "Capital city of Japan"
    weather: "Humid subtropical climate"
```

### 2. Automatic Processing
When an agent is created:
1. **Text Extraction**: Knowledge maps → searchable text chunks
2. **Embedding Generation**: Text chunks → vector embeddings (OpenAI)
3. **Database Storage**: Structured data saved via GORM entities

### 3. Runtime Retrieval
During conversations:
1. **Query Embedding**: User context → query vector
2. **Similarity Search**: Find relevant knowledge via cosine similarity
3. **Context Injection**: Retrieved knowledge added to agent prompt

## Implementation Details

### Knowledge Processing Pipeline
```go
Knowledge Maps → Text Extraction → Embedding → GORM Entity → SQLite Storage
     ↓              ↓               ↓            ↓            ↓
map[string]any → string chunks → []float32 → JSON fields → Database
```

### Search & Retrieval
```go
Query Text → Query Embedding → Similarity Calc → Ranked Results → Context
     ↓            ↓               ↓                ↓             ↓
  string    → []float32     → cosine scores → []string     → Prompt
```

## Configuration

### Memory Service Setup
The service automatically initializes when:
- SQLite is enabled in configuration 
- Memory database path is configured
- OpenAI API key is available for embeddings

### Database Schema
```sql
-- Auto-migrated by GORM
CREATE TABLE knowledge (
    id         INTEGER PRIMARY KEY,
    created_at DATETIME,
    updated_at DATETIME,
    deleted_at DATETIME,
    agent_name TEXT,
    content    TEXT,
    metadata   TEXT, -- JSONB
    embedding  TEXT  -- JSONB
);
```

## Testing

Comprehensive test coverage includes:
- Text extraction with various field types
- Knowledge processing and chunking  
- Cosine similarity calculations
- GORM entity operations
- Graceful degradation scenarios

```bash
go test ./memory/... -v
```

## Benefits Over Previous sqlite-vec Approach

1. **🏗️ Simplified Architecture**: No external dependencies, uses existing GORM setup
2. **🔧 Better Integration**: Seamless integration with existing database infrastructure  
3. **🧪 Easier Testing**: GORM entities support better mocking and testing
4. **⚡ Improved Performance**: In-memory similarity calculations vs external sqlite operations
5. **🛠️ Enhanced Maintainability**: Standard Go patterns, better error handling
6. **🔄 Graceful Degradation**: System works without RAG when embedder unavailable

## Migration Notes

If upgrading from the previous sqlite-vec implementation:
1. No manual setup required - uses existing SQLite database
2. Knowledge automatically migrated via GORM auto-migration
3. Embeddings regenerated on first agent creation with knowledge
4. No breaking changes to agent configuration format 