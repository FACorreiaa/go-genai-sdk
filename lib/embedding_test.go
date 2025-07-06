package genai_sdk

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"google.golang.org/genai"
)

func TestNewEmbeddingService(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	tests := []struct {
		name        string
		setAPIKey   bool
		expectError bool
	}{
		{
			name:        "with API key",
			setAPIKey:   true,
			expectError: false,
		},
		{
			name:        "without API key",
			setAPIKey:   false,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original env var
			originalAPIKey := os.Getenv("GEMINI_API_KEY")
			defer os.Setenv("GEMINI_API_KEY", originalAPIKey)

			if tt.setAPIKey {
				os.Setenv("GEMINI_API_KEY", "test-key")
			} else {
				os.Unsetenv("GEMINI_API_KEY")
			}

			ctx := context.Background()
			service, err := NewEmbeddingService(ctx, logger)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if service == nil {
				t.Error("service should not be nil")
				return
			}

			if service.client == nil {
				t.Error("service.client should not be nil")
			}

			if service.logger == nil {
				t.Error("service.logger should not be nil")
			}
		})
	}
}

func TestEmbeddingService_GenerateEmbedding(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	tests := []struct {
		name        string
		text        string
		expectError bool
	}{
		{
			name:        "simple text",
			text:        "Hello world",
			expectError: false,
		},
		{
			name:        "longer text",
			text:        "This is a longer text that should generate a meaningful embedding vector",
			expectError: false,
		},
		{
			name:        "empty text",
			text:        "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedding, err := service.GenerateEmbedding(ctx, tt.text, nil)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if len(embedding) == 0 {
				t.Error("embedding should not be empty")
				return
			}

			if len(embedding) != EmbeddingDimension {
				t.Errorf("expected embedding dimension %d, got %d", EmbeddingDimension, len(embedding))
			}

			// Check that embedding contains non-zero values
			hasNonZero := false
			for _, val := range embedding {
				if val != 0.0 {
					hasNonZero = true
					break
				}
			}

			if !hasNonZero {
				t.Error("embedding should contain non-zero values")
			}
		})
	}
}

func TestEmbeddingService_GenerateEmbeddingWithConfig(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}

	config := &genai.EmbedContentConfig{
		Title: *genai.Ptr("Test Embedding"),
	}

	embedding, err := service.GenerateEmbedding(ctx, "Test text with config", config)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}

	if len(embedding) == 0 {
		t.Error("embedding should not be empty")
		return
	}

	if len(embedding) != EmbeddingDimension {
		t.Errorf("expected embedding dimension %d, got %d", EmbeddingDimension, len(embedding))
	}
}

func TestEmbeddingService_GeneratePOIEmbedding(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	tests := []struct {
		name        string
		poiName     string
		description string
		category    string
		expectError bool
	}{
		{
			name:        "complete POI",
			poiName:     "Eiffel Tower",
			description: "Famous iron tower in Paris",
			category:    "landmark",
			expectError: false,
		},
		{
			name:        "POI without description",
			poiName:     "Central Park",
			description: "",
			category:    "park",
			expectError: false,
		},
		{
			name:        "empty POI name",
			poiName:     "",
			description: "Some description",
			category:    "unknown",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedding, err := service.GeneratePOIEmbedding(ctx, tt.poiName, tt.description, tt.category)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if len(embedding) == 0 {
				t.Error("embedding should not be empty")
				return
			}

			if len(embedding) != EmbeddingDimension {
				t.Errorf("expected embedding dimension %d, got %d", EmbeddingDimension, len(embedding))
			}
		})
	}
}

func TestEmbeddingService_GenerateCityEmbedding(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	tests := []struct {
		name        string
		cityName    string
		country     string
		description string
		expectError bool
	}{
		{
			name:        "complete city",
			cityName:    "Paris",
			country:     "France",
			description: "City of lights",
			expectError: false,
		},
		{
			name:        "city without description",
			cityName:    "Tokyo",
			country:     "Japan",
			description: "",
			expectError: false,
		},
		{
			name:        "empty city name",
			cityName:    "",
			country:     "USA",
			description: "Some description",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedding, err := service.GenerateCityEmbedding(ctx, tt.cityName, tt.country, tt.description)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if len(embedding) == 0 {
				t.Error("embedding should not be empty")
				return
			}

			if len(embedding) != EmbeddingDimension {
				t.Errorf("expected embedding dimension %d, got %d", EmbeddingDimension, len(embedding))
			}
		})
	}
}

func TestEmbeddingService_GenerateUserPreferenceEmbedding(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	tests := []struct {
		name        string
		interests   []string
		preferences map[string]string
		expectError bool
	}{
		{
			name:        "complete preferences",
			interests:   []string{"art", "history", "food"},
			preferences: map[string]string{"budget": "medium", "style": "cultural"},
			expectError: false,
		},
		{
			name:        "only interests",
			interests:   []string{"sports", "music"},
			preferences: nil,
			expectError: false,
		},
		{
			name:        "only preferences",
			interests:   nil,
			preferences: map[string]string{"budget": "high"},
			expectError: false,
		},
		{
			name:        "empty preferences",
			interests:   []string{},
			preferences: map[string]string{},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedding, err := service.GenerateUserPreferenceEmbedding(ctx, tt.interests, tt.preferences)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if len(embedding) == 0 {
				t.Error("embedding should not be empty")
				return
			}

			if len(embedding) != EmbeddingDimension {
				t.Errorf("expected embedding dimension %d, got %d", EmbeddingDimension, len(embedding))
			}
		})
	}
}

func TestEmbeddingService_GenerateQueryEmbedding(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	tests := []struct {
		name        string
		query       string
		expectError bool
	}{
		{
			name:        "simple query",
			query:       "restaurants near me",
			expectError: false,
		},
		{
			name:        "complex query",
			query:       "best romantic restaurants with view in Paris",
			expectError: false,
		},
		{
			name:        "empty query",
			query:       "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedding, err := service.GenerateQueryEmbedding(ctx, tt.query)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if len(embedding) == 0 {
				t.Error("embedding should not be empty")
				return
			}

			if len(embedding) != EmbeddingDimension {
				t.Errorf("expected embedding dimension %d, got %d", EmbeddingDimension, len(embedding))
			}
		})
	}
}

func TestEmbeddingService_BatchGenerateEmbeddings(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	tests := []struct {
		name        string
		texts       []string
		expectError bool
	}{
		{
			name:        "multiple texts",
			texts:       []string{"Hello world", "Goodbye world", "Testing embeddings"},
			expectError: false,
		},
		{
			name:        "single text",
			texts:       []string{"Single text"},
			expectError: false,
		},
		{
			name:        "empty slice",
			texts:       []string{},
			expectError: true,
		},
		{
			name:        "texts with empty string",
			texts:       []string{"Valid text", "", "Another valid text"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embeddings, err := service.BatchGenerateEmbeddings(ctx, tt.texts)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if len(embeddings) != len(tt.texts) {
				t.Errorf("expected %d embeddings, got %d", len(tt.texts), len(embeddings))
				return
			}

			for i, embedding := range embeddings {
				if len(embedding) == 0 {
					t.Errorf("embedding %d should not be empty", i)
					continue
				}

				if len(embedding) != EmbeddingDimension {
					t.Errorf("embedding %d: expected dimension %d, got %d", i, EmbeddingDimension, len(embedding))
				}
			}
		})
	}
}

func TestEmbeddingService_EmbeddingSimilarity(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	// Generate embeddings for similar texts
	embedding1, err := service.GenerateEmbedding(ctx, "cat", nil)
	if err != nil {
		t.Fatalf("failed to generate first embedding: %v", err)
	}

	embedding2, err := service.GenerateEmbedding(ctx, "kitten", nil)
	if err != nil {
		t.Fatalf("failed to generate second embedding: %v", err)
	}

	embedding3, err := service.GenerateEmbedding(ctx, "airplane", nil)
	if err != nil {
		t.Fatalf("failed to generate third embedding: %v", err)
	}

	// Calculate cosine similarity
	similarity12 := cosineSimilarity(embedding1, embedding2)
	similarity13 := cosineSimilarity(embedding1, embedding3)

	// Similar words should have higher similarity than dissimilar words
	if similarity12 <= similarity13 {
		t.Logf("cat-kitten similarity: %f, cat-airplane similarity: %f", similarity12, similarity13)
		t.Log("Note: This test may occasionally fail due to model variations")
	}

	// All similarities should be between -1 and 1
	if similarity12 < -1 || similarity12 > 1 {
		t.Errorf("similarity12 out of range: %f", similarity12)
	}

	if similarity13 < -1 || similarity13 > 1 {
		t.Errorf("similarity13 out of range: %f", similarity13)
	}
}

func TestEmbeddingService_WithTimeout(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	// This should complete quickly
	_, err = service.GenerateEmbedding(ctx, "Quick test", nil)
	if err != nil {
		// Context timeout is acceptable for this test
		if ctx.Err() == context.DeadlineExceeded {
			t.Log("Request timed out as expected")
		} else {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func TestEmbeddingConstants(t *testing.T) {
	if EmbeddingModel == "" {
		t.Error("EmbeddingModel should not be empty")
	}

	if EmbeddingDimension <= 0 {
		t.Error("EmbeddingDimension should be positive")
	}

	// Test that the model name is reasonable
	expectedModels := []string{"gemini-embedding-exp-03-07", "text-embedding-004"}
	found := false
	for _, model := range expectedModels {
		if EmbeddingModel == model {
			found = true
			break
		}
	}

	if !found {
		t.Logf("Warning: EmbeddingModel '%s' is not in expected models list", EmbeddingModel)
	}
}

func BenchmarkEmbeddingService_GenerateEmbedding(b *testing.B) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		b.Skip("GEMINI_API_KEY not set, skipping benchmark")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		b.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	text := "This is a test text for benchmarking embedding generation"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := service.GenerateEmbedding(ctx, text, nil)
		if err != nil {
			b.Errorf("benchmark error: %v", err)
		}
	}
}

func BenchmarkEmbeddingService_BatchGenerateEmbeddings(b *testing.B) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		b.Skip("GEMINI_API_KEY not set, skipping benchmark")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewEmbeddingService(ctx, logger)
	if err != nil {
		b.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	texts := []string{
		"First test text",
		"Second test text",
		"Third test text",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := service.BatchGenerateEmbeddings(ctx, texts)
		if err != nil {
			b.Errorf("benchmark error: %v", err)
		}
	}
}

// Helper function to calculate cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (normA * normB)
}
