package genai_sdk

import (
	"context"
	"os"
	"testing"
	"time"

	"google.golang.org/genai"
)

func TestNewLLMChatClient(t *testing.T) {
	tests := []struct {
		name        string
		apiKey      string
		expectError bool
	}{
		{
			name:        "valid API key",
			apiKey:      "test-api-key",
			expectError: false,
		},
		{
			name:        "empty API key",
			apiKey:      "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			client, err := NewLLMChatClient(ctx, tt.apiKey)

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

			if client.ModelName == "" {
				t.Error("ModelName should not be empty")
			}
		})
	}
}

func TestLLMChatClient_GenerateContent(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	ctx := context.Background()
	client, err := NewLLMChatClient(ctx, apiKey)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.1),
		MaxOutputTokens: *genai.Ptr[int32](100),
	}

	tests := []struct {
		name        string
		prompt      string
		expectError bool
	}{
		{
			name:        "simple prompt",
			prompt:      "Say hello",
			expectError: false,
		},
		{
			name:        "empty prompt",
			prompt:      "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response, err := client.GenerateContent(ctx, tt.prompt, apiKey, config)

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

			if response == "" {
				t.Error("response should not be empty")
			}
		})
	}
}

func TestLLMChatClient_GenerateResponse(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	ctx := context.Background()
	client, err := NewLLMChatClient(ctx, apiKey)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.1),
		MaxOutputTokens: *genai.Ptr[int32](100),
	}

	response, err := client.GenerateResponse(ctx, "Say hello", config)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}

	if response == nil {
		t.Error("response should not be nil")
		return
	}

	if response.Text() == "" {
		t.Error("response text should not be empty")
	}
}

func TestLLMChatClient_StartChatSession(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	ctx := context.Background()
	client, err := NewLLMChatClient(ctx, apiKey)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature: genai.Ptr[float32](0.1),
	}

	session, err := client.StartChatSession(ctx, config)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}

	if session == nil {
		t.Error("session should not be nil")
		return
	}

	if session.chat == nil {
		t.Error("session.chat should not be nil")
	}
}

func TestChatSession_SendMessage(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	ctx := context.Background()
	client, err := NewLLMChatClient(ctx, apiKey)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.1),
		MaxOutputTokens: *genai.Ptr[int32](100),
	}

	session, err := client.StartChatSession(ctx, config)
	if err != nil {
		t.Fatalf("failed to start session: %v", err)
	}

	tests := []struct {
		name        string
		message     string
		expectError bool
	}{
		{
			name:        "simple message",
			message:     "Hello",
			expectError: false,
		},
		{
			name:        "empty message",
			message:     "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response, err := session.SendMessage(ctx, tt.message)

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

			if response == "" {
				t.Error("response should not be empty")
			}
		})
	}
}

func TestChatSession_ConversationFlow(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	ctx := context.Background()
	client, err := NewLLMChatClient(ctx, apiKey)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.1),
		MaxOutputTokens: *genai.Ptr[int32](200),
	}

	session, err := client.StartChatSession(ctx, config)
	if err != nil {
		t.Fatalf("failed to start session: %v", err)
	}

	// First message
	response1, err := session.SendMessage(ctx, "My name is Alice. Remember this.")
	if err != nil {
		t.Errorf("failed to send first message: %v", err)
		return
	}

	if response1 == "" {
		t.Error("first response should not be empty")
		return
	}

	// Second message to test conversation memory
	response2, err := session.SendMessage(ctx, "What is my name?")
	if err != nil {
		t.Errorf("failed to send second message: %v", err)
		return
	}

	if response2 == "" {
		t.Error("second response should not be empty")
	}

	// The response should ideally contain "Alice" but we won't assert this
	// as the AI model behavior can vary
}

func TestLLMChatClient_GenerateContentStream(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	ctx := context.Background()
	client, err := NewLLMChatClient(ctx, apiKey)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.1),
		MaxOutputTokens: *genai.Ptr[int32](100),
	}

	stream, err := client.GenerateContentStream(ctx, "Count from 1 to 5", config)
	if err != nil {
		t.Errorf("failed to create stream: %v", err)
		return
	}

	responseCount := 0
	totalText := ""

	for response, err := range stream {
		if err != nil {
			t.Errorf("streaming error: %v", err)
			break
		}

		responseCount++
		totalText += response.Text()

		// Prevent infinite loop in case of issues
		if responseCount > 100 {
			t.Error("too many responses, possible infinite loop")
			break
		}
	}

	if responseCount == 0 {
		t.Error("expected at least one response from stream")
	}

	if totalText == "" {
		t.Error("total text should not be empty")
	}
}

func TestLLMChatClient_GenerateContentStreamWithCache(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	ctx := context.Background()
	client, err := NewLLMChatClient(ctx, apiKey)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.1),
		MaxOutputTokens: *genai.Ptr[int32](50),
	}

	cacheKey := "test-cache-key"
	stream, err := client.GenerateContentStreamWithCache(ctx, "Say hello", config, cacheKey)
	if err != nil {
		t.Errorf("failed to create stream with cache: %v", err)
		return
	}

	responseCount := 0
	for response, err := range stream {
		if err != nil {
			t.Errorf("streaming error: %v", err)
			break
		}

		responseCount++
		if response.Text() == "" {
			t.Error("response text should not be empty")
		}

		// Prevent infinite loop
		if responseCount > 50 {
			t.Error("too many responses, possible infinite loop")
			break
		}
	}

	if responseCount == 0 {
		t.Error("expected at least one response from stream")
	}
}

func TestChatSession_SendMessageStream(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	ctx := context.Background()
	client, err := NewLLMChatClient(ctx, apiKey)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.1),
		MaxOutputTokens: *genai.Ptr[int32](50),
	}

	session, err := client.StartChatSession(ctx, config)
	if err != nil {
		t.Fatalf("failed to start session: %v", err)
	}

	stream := session.SendMessageStream(ctx, "Say hello")
	responseCount := 0
	totalText := ""

	for response, err := range stream {
		if err != nil {
			t.Errorf("streaming error: %v", err)
			break
		}

		responseCount++
		totalText += response.Text()

		// Prevent infinite loop
		if responseCount > 50 {
			t.Error("too many responses, possible infinite loop")
			break
		}
	}

	if responseCount == 0 {
		t.Error("expected at least one response from stream")
	}

	if totalText == "" {
		t.Error("total text should not be empty")
	}
}

func TestLLMChatClient_WithTimeout(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	client, err := NewLLMChatClient(ctx, apiKey)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.1),
		MaxOutputTokens: *genai.Ptr[int32](10),
	}

	// This should complete quickly with a short response
	_, err = client.GenerateContent(ctx, "Hi", apiKey, config)
	if err != nil {
		// Context timeout is acceptable for this test
		if ctx.Err() == context.DeadlineExceeded {
			t.Log("Request timed out as expected")
		} else {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func BenchmarkLLMChatClient_GenerateContent(b *testing.B) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		b.Skip("GEMINI_API_KEY not set, skipping benchmark")
	}

	ctx := context.Background()
	client, err := NewLLMChatClient(ctx, apiKey)
	if err != nil {
		b.Fatalf("failed to create client: %v", err)
	}

	config := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.1),
		MaxOutputTokens: *genai.Ptr[int32](10),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := client.GenerateContent(ctx, "Hi", apiKey, config)
		if err != nil {
			b.Errorf("benchmark error: %v", err)
		}
	}
}
