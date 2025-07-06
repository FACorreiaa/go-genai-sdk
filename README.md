# Go Genai SDK

A Go SDK for Google's Gemini AI API, providing easy-to-use interfaces for chat functionality and embeddings generation.

## Features

- **Chat API**: Generate responses, create chat sessions, and stream responses
- **Embeddings API**: Generate embeddings for text, POIs, cities, and user preferences
- **Streaming Support**: Real-time response streaming
- **Batch Processing**: Generate multiple embeddings in batches
- **OpenTelemetry Integration**: Built-in observability and tracing
- **Comprehensive Testing**: Full test coverage with integration tests

## Installation

```bash
go get github.com/FACorreiaa/go-genai-sdk
```

## Prerequisites

- Go 1.24.4 or higher
- Gemini API key (set as `GEMINI_API_KEY` environment variable)

## Quick Start

### Chat Example

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    
    genai_sdk "github.com/FACorreiaa/go-genai-sdk"
    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()
    apiKey := os.Getenv("GEMINI_API_KEY")
    
    client, err := genai_sdk.NewLLMChatClient(ctx, apiKey)
    if err != nil {
        log.Fatal(err)
    }
    
    config := &genai.GenerateContentConfig{
        Temperature: genai.Ptr[float32](0.7),
        MaxOutputTokens: genai.Ptr[int32](1000),
    }
    
    response, err := client.GenerateContent(ctx, "Hello! Tell me a joke.", apiKey, config)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Response: %s\n", response)
}
```

### Embedding Example

```go
package main

import (
    "context"
    "fmt"
    "log"
    "log/slog"
    "os"
    
    genai_sdk "github.com/FACorreiaa/go-genai-sdk"
)

func main() {
    ctx := context.Background()
    logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
    
    service, err := genai_sdk.NewEmbeddingService(ctx, logger)
    if err != nil {
        log.Fatal(err)
    }
    defer service.Close()
    
    embedding, err := service.GenerateEmbedding(ctx, "Hello world", nil)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Generated embedding with %d dimensions\n", len(embedding))
}
```

## API Reference

### Chat Client

#### `NewLLMChatClient(ctx context.Context, apiKey string) (*LLMChatClient, error)`

Creates a new chat client with the provided API key.

#### `GenerateContent(ctx context.Context, prompt, apiKey string, config *genai.GenerateContentConfig) (string, error)`

Generates a single response for the given prompt.

#### `StartChatSession(ctx context.Context, config *genai.GenerateContentConfig) (*ChatSession, error)`

Creates a new chat session for maintaining conversation context.

#### `GenerateContentStream(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (iter.Seq2[*genai.GenerateContentResponse, error], error)`

Generates streaming responses for real-time interactions.

### Embedding Service

#### `NewEmbeddingService(ctx context.Context, logger *slog.Logger) (*EmbeddingService, error)`

Creates a new embedding service instance.

#### `GenerateEmbedding(ctx context.Context, text string, config *genai.EmbedContentConfig) ([]float32, error)`

Generates an embedding vector for the given text.

#### `GeneratePOIEmbedding(ctx context.Context, name, description, category string) ([]float32, error)`

Generates embeddings specifically for Point of Interest data.

#### `GenerateCityEmbedding(ctx context.Context, name, country, description string) ([]float32, error)`

Generates embeddings for city information.

#### `BatchGenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error)`

Generates embeddings for multiple texts in a single batch.

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Model Configuration

The SDK uses the following default models:
- Chat: `gemini-2.0-flash`
- Embeddings: `gemini-embedding-exp-03-07`

## Testing

Run all tests:
```bash
go test ./...
```

Run tests with integration tests (requires API key):
```bash
GEMINI_API_KEY=your_api_key go test ./...
```

Run benchmarks:
```bash
GEMINI_API_KEY=your_api_key go test -bench=. ./...
```

## Examples

See `main.go` for comprehensive usage examples including:
- Basic chat functionality
- Chat sessions with conversation memory
- Streaming responses
- Various embedding types (text, POI, city, user preferences)
- Batch embedding generation

## Error Handling

The SDK provides comprehensive error handling with OpenTelemetry tracing. All errors are wrapped with contextual information to help with debugging.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please create an issue in the GitHub repository.
