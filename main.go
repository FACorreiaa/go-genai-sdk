package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"

	genai_sdk "github.com/FACorreiaa/go-genai-sdk/lib"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	// Get API key from environment
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY environment variable is required")
	}

	// Example 1: Basic chat functionality
	fmt.Println("=== Basic Chat Example ===")
	if err := basicChatExample(ctx, apiKey); err != nil {
		log.Printf("Basic chat example failed: %v", err)
	}

	// Example 2: Chat session
	fmt.Println("\n=== Chat Session Example ===")
	if err := chatSessionExample(ctx, apiKey); err != nil {
		log.Printf("Chat session example failed: %v", err)
	}

	// Example 3: Streaming chat
	fmt.Println("\n=== Streaming Chat Example ===")
	if err := streamingChatExample(ctx, apiKey); err != nil {
		log.Printf("Streaming chat example failed: %v", err)
	}

	// Example 4: Basic embedding
	fmt.Println("\n=== Basic Embedding Example ===")
	if err := basicEmbeddingExample(ctx, logger); err != nil {
		log.Printf("Basic embedding example failed: %v", err)
	}

	// Example 5: POI embedding
	fmt.Println("\n=== POI Embedding Example ===")
	if err := poiEmbeddingExample(ctx, logger); err != nil {
		log.Printf("POI embedding example failed: %v", err)
	}

	// Example 6: Batch embeddings
	fmt.Println("\n=== Batch Embedding Example ===")
	if err := batchEmbeddingExample(ctx, logger); err != nil {
		log.Printf("Batch embedding example failed: %v", err)
	}
}

func basicChatExample(ctx context.Context, apiKey string) error {
	client, err := genai_sdk.NewLLMChatClient(ctx, apiKey)
	if err != nil {
		return fmt.Errorf("failed to create chat client: %w", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.7),
		MaxOutputTokens: *genai.Ptr[int32](1000),
	}

	response, err := client.GenerateContent(ctx, "Hello! Tell me a short joke.", apiKey, config)
	if err != nil {
		return fmt.Errorf("failed to generate content: %w", err)
	}

	fmt.Printf("Response: %s\n", response)
	return nil
}

func chatSessionExample(ctx context.Context, apiKey string) error {
	client, err := genai_sdk.NewLLMChatClient(ctx, apiKey)
	if err != nil {
		return fmt.Errorf("failed to create chat client: %w", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature: genai.Ptr[float32](0.5),
	}

	session, err := client.StartChatSession(ctx, config)
	if err != nil {
		return fmt.Errorf("failed to start chat session: %w", err)
	}

	// Send first message
	response1, err := session.SendMessage(ctx, "My name is Alice. What's yours?")
	if err != nil {
		return fmt.Errorf("failed to send first message: %w", err)
	}
	fmt.Printf("First response: %s\n", response1)

	// Send follow-up message
	response2, err := session.SendMessage(ctx, "What's my name?")
	if err != nil {
		return fmt.Errorf("failed to send follow-up message: %w", err)
	}
	fmt.Printf("Follow-up response: %s\n", response2)

	return nil
}

func streamingChatExample(ctx context.Context, apiKey string) error {
	client, err := genai_sdk.NewLLMChatClient(ctx, apiKey)
	if err != nil {
		return fmt.Errorf("failed to create chat client: %w", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature: genai.Ptr[float32](0.7),
	}

	stream, err := client.GenerateContentStream(ctx, "Tell me a short story about AI", config)
	if err != nil {
		return fmt.Errorf("failed to create stream: %w", err)
	}

	fmt.Print("Streaming response: ")
	for response, err := range stream {
		if err != nil {
			return fmt.Errorf("streaming error: %w", err)
		}
		fmt.Print(response.Text())
	}
	fmt.Println()

	return nil
}

func basicEmbeddingExample(ctx context.Context, logger *slog.Logger) error {
	service, err := genai_sdk.NewEmbeddingService(ctx, logger)
	if err != nil {
		return fmt.Errorf("failed to create embedding service: %w", err)
	}

	text := "This is a sample text for embedding generation"
	embedding, err := service.GenerateEmbedding(ctx, text, nil)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	fmt.Printf("Generated embedding with %d dimensions\n", len(embedding))
	fmt.Printf("First 5 values: %v\n", embedding[:5])

	return nil
}

func poiEmbeddingExample(ctx context.Context, logger *slog.Logger) error {
	service, err := genai_sdk.NewEmbeddingService(ctx, logger)
	if err != nil {
		return fmt.Errorf("failed to create embedding service: %w", err)
	}

	embedding, err := service.GeneratePOIEmbedding(ctx, "Eiffel Tower", "Famous iron tower in Paris", "landmark")
	if err != nil {
		return fmt.Errorf("failed to generate POI embedding: %w", err)
	}

	fmt.Printf("Generated POI embedding with %d dimensions\n", len(embedding))
	fmt.Printf("First 5 values: %v\n", embedding[:5])

	return nil
}

func batchEmbeddingExample(ctx context.Context, logger *slog.Logger) error {
	service, err := genai_sdk.NewEmbeddingService(ctx, logger)
	if err != nil {
		return fmt.Errorf("failed to create embedding service: %w", err)
	}

	texts := []string{
		"Paris is the capital of France",
		"Tokyo is known for its technology",
		"New York is famous for its skyline",
	}

	embeddings, err := service.BatchGenerateEmbeddings(ctx, texts)
	if err != nil {
		return fmt.Errorf("failed to generate batch embeddings: %w", err)
	}

	fmt.Printf("Generated %d embeddings\n", len(embeddings))
	for i, embedding := range embeddings {
		fmt.Printf("Text %d: %d dimensions, first 3 values: %v\n", i+1, len(embedding), embedding[:3])
	}

	return nil
}
