# Go Genai SDK

A thin Go wrapper around [`google.golang.org/genai`](https://github.com/googleapis/go-genai) for Gemini chat, streaming, embeddings, and response normalization.

## Installation

```bash
go get github.com/FACorreiaa/go-genai-sdk/v2
```

## Quick Start

```go
ctx := context.Background()
client, err := genai_sdk.NewGeminiChatClient(ctx, os.Getenv("GEMINI_API_KEY"), "gemini-2.5-flash")
if err != nil {
    log.Fatal(err)
}
defer client.Close()

text, err := client.GenerateText(ctx, "Say hello", &genai.GenerateContentConfig{
    Temperature: genai.Ptr[float32](0.2),
})
```

## v2 ChatClient

| Method | Purpose |
|--------|---------|
| `Generate` | Full `GenerateContentResponse` with retry |
| `GenerateText` | `Generate` + `ExtractText` |
| `GenerateStream` | Streaming iterator with retry on init |
| `Close` | Client cleanup hook |

## Response helpers

```go
clean := genai_sdk.CleanJSON(raw)       // strips null, fences, section tags
text, err := genai_sdk.ExtractText(resp)
p, c, t := genai_sdk.ExtractUsage(resp)
```

## Embeddings

```go
embed, err := genai_sdk.NewGeminiEmbeddingClient(ctx, apiKey, "gemini-embedding-exp-03-07", logger)
vec, err := embed.GeneratePOIEmbedding(ctx, name, description, category)
```

## Testing

```bash
go test ./lib
GEMINI_API_KEY=... go test ./lib -count=1
```