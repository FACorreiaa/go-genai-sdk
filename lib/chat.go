package genai_sdk

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
	"os"

	"google.golang.org/genai"
)

// ChatClient abstracts LLM chat capabilities needed by domain services.
type ChatClient interface {
	GenerateResponse(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error)
	GenerateContent(ctx context.Context, prompt, apiKey string, config *genai.GenerateContentConfig) (string, error)
	GenerateContentStream(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (iter.Seq2[*genai.GenerateContentResponse, error], error)
	GenerateContentStreamWithCache(ctx context.Context, prompt string, config *genai.GenerateContentConfig, cacheKey string) (iter.Seq2[*genai.GenerateContentResponse, error], error)
	Model() string
	StartChatSession(ctx context.Context, config *genai.GenerateContentConfig) (*ChatSession, error)
}

// GeminiChatClient adapts the generativeAI LLM client to the ChatClient interface.
type GeminiChatClient struct {
	client *genai.Client
	model  string
}

// NewGeminiChatClient creates a ChatClient backed by Gemini.
func NewGeminiChatClient(ctx context.Context, apiKey, modelName string) (ChatClient, error) {
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if modelName == "" {
		modelName = "gemini-1.5-flash"
	}
	client, err := genai.NewClient(ctx, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		return nil, err
	}
	return &GeminiChatClient{client: client, model: modelName}, nil
}

func (g *GeminiChatClient) GenerateResponse(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
	resp, err := g.client.Models.GenerateContent(ctx, g.model, genai.Text(prompt), config)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func (g *GeminiChatClient) GenerateContent(ctx context.Context, prompt, apiKey string, config *genai.GenerateContentConfig) (string, error) {
	// Note: apiKey argument is ignored as the client is already initialized with one.
	// Function signature kept for interface compatibility if needed, but we rely on the client's key.
	resp, err := g.client.Models.GenerateContent(ctx, g.model, genai.Text(prompt), config)
	if err != nil {
		return "", err
	}
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		return resp.Candidates[0].Content.Parts[0].Text, nil
	}
	err = fmt.Errorf("no content generated")
	return "", err
}

func (g *GeminiChatClient) GenerateContentStream(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (iter.Seq2[*genai.GenerateContentResponse, error], error) {
	// The SDK returns the iterator directly.
	resp := g.client.Models.GenerateContentStream(ctx, g.model, genai.Text(prompt), config)
	return resp, nil
}

func (g *GeminiChatClient) GenerateContentStreamWithCache(ctx context.Context, prompt string, config *genai.GenerateContentConfig, cacheKey string) (iter.Seq2[*genai.GenerateContentResponse, error], error) {
	// Fallback to normal stream for now as in original logic, but we keep the method signature.
	if cacheKey != "" {
		slog.InfoContext(ctx, "Cache key provided but currently ignored in direct implementation", "cacheKey", cacheKey)
	}

	resp := g.client.Models.GenerateContentStream(ctx, g.model, genai.Text(prompt), config)
	return resp, nil
}

func (g *GeminiChatClient) Model() string {
	return g.model
}

type ChatSession struct {
	chat *genai.Chat
}

func (g *GeminiChatClient) StartChatSession(ctx context.Context, config *genai.GenerateContentConfig) (*ChatSession, error) {
	chat, err := g.client.Chats.Create(ctx, g.model, config, nil)
	if err != nil {
		return nil, err
	}

	return &ChatSession{chat: chat}, nil
}

func (cs *ChatSession) SendMessage(ctx context.Context, message string) (string, error) {
	result, err := cs.chat.SendMessage(ctx, genai.Part{Text: message})
	if err != nil {
		return "", err
	}

	if len(result.Candidates) > 0 && len(result.Candidates[0].Content.Parts) > 0 {
		responseText := result.Candidates[0].Content.Parts[0].Text
		return responseText, nil
	}
	return "", fmt.Errorf("no response content")
}

func (cs *ChatSession) SendMessageStream(ctx context.Context, message string) iter.Seq2[*genai.GenerateContentResponse, error] {
	return cs.chat.SendMessageStream(ctx, genai.Part{Text: message})
}
