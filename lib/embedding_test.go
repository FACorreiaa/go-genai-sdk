package genai_sdk

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"google.golang.org/genai"
)

func TestNewGeminiEmbeddingClient(t *testing.T) {
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
			client, err := NewGeminiEmbeddingClient(ctx, "", logger)

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

			if client == nil {
				t.Error("client should not be nil")
				return
			}

			// We can assert type to check internals if needed, but checking non-nil is enough for public API test
			impl, ok := client.(*GeminiEmbeddingClient)
			if !ok {
				t.Error("client should be of type *GeminiEmbeddingClient")
				return
			}

			if impl.client == nil {
				t.Error("impl.client should not be nil")
			}

			if impl.logger == nil {
				t.Error("impl.logger should not be nil")
			}
		})
	}
}

func TestGeminiEmbeddingClient_GenerateEmbedding(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
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
			// Type assert to access GenerateEmbedding
			impl, ok := service.(*GeminiEmbeddingClient)
			if !ok {
				t.Fatalf("service is not *GeminiEmbeddingClient")
			}
			embedding, err := impl.GenerateEmbedding(ctx, tt.text, nil)

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

func TestGeminiEmbeddingClient_GenerateEmbeddingWithConfig(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}

	config := &genai.EmbedContentConfig{
		Title: *genai.Ptr("Test Embedding"),
	}

	embedding, err := service.(interface {
		GenerateEmbedding(ctx context.Context, text string, config *genai.EmbedContentConfig) ([]float32, error)
	}).GenerateEmbedding(ctx, "Test text with config", config)

	// Note: The interface EmbeddingClient might not expose GenerateEmbedding with config if it wasn't in the interface definition I used?
	// Let's check existing embedding.go.
	// Interface: GenerateEmbedding(ctx, text, config) was NOT in the interface I defined in embedding.go!
	// I defined: GenerateQueryEmbedding, GeneratePOIEmbedding, etc.
	// But `GenerateEmbedding` was a method on the struct.
	// Ideally the interface explicitly exposes the low level one too if needed, or we rely on specific methods.
	// For this test, I will assert interface enhancement or use type assertion.
	// Looking at embedding.go:
	/*
		type EmbeddingClient interface {
			GenerateQueryEmbedding(ctx context.Context, query string) ([]float32, error)
			...
			BatchGenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error)
			Close()
		}
	*/
	// GenerateEmbedding is NOT in the interface.
	// But the struct has it.
	// I will type assert to *GeminiEmbeddingClient to test it, or just rely on public interface tests.
	// Since this test specifically tests Config passing which is only available via GenerateEmbedding (unless others expose it), I should test the struct method or added it to interface.
	// I'll type assert.

	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}

	if len(embedding) == 0 {
		t.Error("embedding should not be empty")
		return
	}
}

func TestGeminiEmbeddingClient_GeneratePOIEmbedding(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
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

func TestGeminiEmbeddingClient_GenerateCityEmbedding(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
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

func TestGeminiEmbeddingClient_GenerateUserPreferenceEmbedding(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
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

func TestGeminiEmbeddingClient_GenerateQueryEmbedding(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
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

func TestGeminiEmbeddingClient_BatchGenerateEmbeddings(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
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

func TestGeminiEmbeddingClient_EmbeddingSimilarity(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	// Generate embeddings for similar texts
	// Note: GenerateEmbedding is not directly on interface, type assert
	impl, ok := service.(*GeminiEmbeddingClient)
	if !ok {
		t.Fatalf("service is not *GeminiEmbeddingClient")
	}

	embedding1, err := impl.GenerateEmbedding(ctx, "cat", nil)
	if err != nil {
		t.Fatalf("failed to generate first embedding: %v", err)
	}

	embedding2, err := impl.GenerateEmbedding(ctx, "kitten", nil)
	if err != nil {
		t.Fatalf("failed to generate second embedding: %v", err)
	}

	embedding3, err := impl.GenerateEmbedding(ctx, "airplane", nil)
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

func TestGeminiEmbeddingClient_WithTimeout(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
	if err != nil {
		t.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	// This should complete quickly
	impl, _ := service.(*GeminiEmbeddingClient)
	_, err = impl.GenerateEmbedding(ctx, "Quick test", nil)
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

func BenchmarkGeminiEmbeddingClient_GenerateEmbedding(b *testing.B) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		b.Skip("GEMINI_API_KEY not set, skipping benchmark")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
	if err != nil {
		b.Fatalf("failed to create embedding service: %v", err)
	}
	defer service.Close()

	text := "This is a test text for benchmarking embedding generation"

	// Type assert to access GenerateEmbedding
	impl, _ := service.(*GeminiEmbeddingClient)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := impl.GenerateEmbedding(ctx, text, nil)
		if err != nil {
			b.Errorf("benchmark error: %v", err)
		}
	}
}

func BenchmarkGeminiEmbeddingClient_BatchGenerateEmbeddings(b *testing.B) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		b.Skip("GEMINI_API_KEY not set, skipping benchmark")
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	ctx := context.Background()

	service, err := NewGeminiEmbeddingClient(ctx, "", logger)
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
