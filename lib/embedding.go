package genai_sdk

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"google.golang.org/genai"
)

const (
	// Gemini embedding model - using the latest embedding model
	//EmbeddingModel = "text-embedding-004"
	EmbeddingModel = "gemini-embedding-exp-03-07"
	// Standard embedding dimension for Gemini text-embedding-004
	EmbeddingDimension = 768
)

// EmbeddingClient abstracts embedding operations needed by domain services.
type EmbeddingClient interface {
	GenerateQueryEmbedding(ctx context.Context, query string) ([]float32, error)
	GeneratePOIEmbedding(ctx context.Context, name, description, category string) ([]float32, error)
	GenerateCityEmbedding(ctx context.Context, name, country, description string) ([]float32, error)
	GenerateUserPreferenceEmbedding(ctx context.Context, interests []string, preferences map[string]string) ([]float32, error)
	BatchGenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error)
	Close()
}

// GeminiEmbeddingClient adapts the generativeAI embedding service.
type GeminiEmbeddingClient struct {
	client *genai.Client
	logger *slog.Logger
}

// NewGeminiEmbeddingClient creates an EmbeddingClient backed by Gemini.
func NewGeminiEmbeddingClient(ctx context.Context, apiKey, modelName string, logger *slog.Logger) (EmbeddingClient, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}
	if modelName == "" {
		modelName = EmbeddingModel
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	return &GeminiEmbeddingClient{
		client: client,
		logger: logger,
	}, nil
}

// Close provides a noop closer to align with consumers expecting a cleanup hook.
func (es *GeminiEmbeddingClient) Close() {
	if es == nil {
		return
	}
}

// GenerateEmbedding generates an embedding vector for the given text
func (es *GeminiEmbeddingClient) GenerateEmbedding(ctx context.Context, text string, config *genai.EmbedContentConfig) ([]float32, error) {
	if text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Use the embedding model to generate embeddings
	embedding, err := es.client.Models.EmbedContent(ctx, EmbeddingModel, genai.Text(text), config)
	if err != nil {
		es.logger.ErrorContext(ctx, "Failed to generate embedding",
			slog.Any("error", err),
			slog.String("text_preview", text[:min(100, len(text))]))
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Extract the embedding values
	if embedding == nil || len(embedding.Embeddings) == 0 {
		return nil, fmt.Errorf("received empty embedding from API")
	}

	// Get the first embedding (assuming single text input)
	contentEmbedding := embedding.Embeddings[0]
	if contentEmbedding == nil || len(contentEmbedding.Values) == 0 {
		return nil, fmt.Errorf("received empty embedding values from API")
	}

	es.logger.DebugContext(ctx, "Embedding generated",
		slog.Int("dimension", len(contentEmbedding.Values)),
		slog.String("model", EmbeddingModel))

	return contentEmbedding.Values, nil
}

// GeneratePOIEmbedding generates an embedding specifically for POI data
func (es *GeminiEmbeddingClient) GeneratePOIEmbedding(ctx context.Context, name, description, category string) ([]float32, error) {
	if strings.TrimSpace(name) == "" {
		return nil, fmt.Errorf("poi name cannot be empty")
	}

	// Create a comprehensive text representation of the POI
	var text string
	if description != "" {
		text = fmt.Sprintf("Name: %s\nCategory: %s\nDescription: %s", name, category, description)
	} else {
		text = fmt.Sprintf("Name: %s\nCategory: %s", name, category)
	}

	embedding, err := es.GenerateEmbedding(ctx, text, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate POI embedding: %w", err)
	}

	return embedding, nil
}

// GenerateCityEmbedding generates an embedding specifically for city data
func (es *GeminiEmbeddingClient) GenerateCityEmbedding(ctx context.Context, name, country, description string) ([]float32, error) {
	if strings.TrimSpace(name) == "" {
		return nil, fmt.Errorf("city name cannot be empty")
	}

	// Create a comprehensive text representation of the city
	text := fmt.Sprintf("City: %s, Country: %s", name, country)
	if description != "" {
		text += fmt.Sprintf("\nDescription: %s", description)
	}

	embedding, err := es.GenerateEmbedding(ctx, text, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate city embedding: %w", err)
	}

	return embedding, nil
}

// GenerateUserPreferenceEmbedding generates an embedding for user preferences
func (es *GeminiEmbeddingClient) GenerateUserPreferenceEmbedding(ctx context.Context, interests []string, preferences map[string]string) ([]float32, error) {
	// Create a text representation of user preferences
	text := "User Interests: "
	for i, interest := range interests {
		if i > 0 {
			text += ", "
		}
		text += interest
	}

	if len(preferences) > 0 {
		text += "\nPreferences: "
		for key, value := range preferences {
			text += fmt.Sprintf("%s: %s; ", key, value)
		}
	}

	embedding, err := es.GenerateEmbedding(ctx, text, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate user preference embedding: %w", err)
	}

	return embedding, nil
}

// GenerateQueryEmbedding generates an embedding for search queries
func (es *GeminiEmbeddingClient) GenerateQueryEmbedding(ctx context.Context, query string) ([]float32, error) {
	embedding, err := es.GenerateEmbedding(ctx, query, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	return embedding, nil
}

// BatchGenerateEmbeddings generates embeddings for multiple texts at once
func (es *GeminiEmbeddingClient) BatchGenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided for batch embedding")
	}

	embeddings := make([][]float32, len(texts))
	var err error

	// Generate embeddings sequentially
	// TODO: Implement concurrent processing with rate limiting if needed
	for i, text := range texts {
		embeddings[i], err = es.GenerateEmbedding(ctx, text, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding for text at index %d: %w", i, err)
		}
	}

	es.logger.InfoContext(ctx, "Batch embeddings generated",
		slog.Int("count", len(embeddings)),
		slog.String("model", EmbeddingModel))

	return embeddings, nil
}
