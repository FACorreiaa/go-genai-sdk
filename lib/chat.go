package genai_sdk

import (
	"context"
	"fmt"
	"iter"
	"log/slog"

	"google.golang.org/genai"
)

// ChatClient abstracts LLM chat capabilities needed by domain services.
type ChatClient interface {
	Generate(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error)
	GenerateText(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (string, error)
	GenerateStream(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (iter.Seq2[*genai.GenerateContentResponse, error], error)
	Model() string
	Close() error
	StartChatSession(ctx context.Context, config *genai.GenerateContentConfig) (*ChatSession, error)
}

// GeminiChatClient adapts the Gemini client to the ChatClient interface.
type GeminiChatClient struct {
	client      *genai.Client
	model       string
	retryPolicy RetryPolicy
	logger      *slog.Logger
}

// NewGeminiChatClient creates a ChatClient backed by Gemini.
func NewGeminiChatClient(ctx context.Context, apiKey, modelName string) (ChatClient, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}
	if modelName == "" {
		return nil, fmt.Errorf("model name is required")
	}
	client, err := genai.NewClient(ctx, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		return nil, err
	}
	return &GeminiChatClient{
		client:      client,
		model:       modelName,
		retryPolicy: DefaultRetryPolicy,
		logger:      slog.Default(),
	}, nil
}

// WithRetryPolicy overrides the default retry policy.
func (g *GeminiChatClient) WithRetryPolicy(policy RetryPolicy) *GeminiChatClient {
	g.retryPolicy = policy
	return g
}

// WithLogger sets the logger used for retry diagnostics.
func (g *GeminiChatClient) WithLogger(logger *slog.Logger) *GeminiChatClient {
	if logger != nil {
		g.logger = logger
	}
	return g
}

func (g *GeminiChatClient) Generate(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
	return retryWithBackoff(ctx, g.retryPolicy, g.logger, "Generate",
		func() (*genai.GenerateContentResponse, error) {
			return g.client.Models.GenerateContent(ctx, g.model, genai.Text(prompt), config)
		})
}

func (g *GeminiChatClient) GenerateText(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (string, error) {
	resp, err := g.Generate(ctx, prompt, config)
	if err != nil {
		return "", err
	}
	return ExtractText(resp)
}

func (g *GeminiChatClient) GenerateStream(ctx context.Context, prompt string, config *genai.GenerateContentConfig) (iter.Seq2[*genai.GenerateContentResponse, error], error) {
	return retryWithBackoff(ctx, g.retryPolicy, g.logger, "GenerateStream",
		func() (iter.Seq2[*genai.GenerateContentResponse, error], error) {
			stream := g.client.Models.GenerateContentStream(ctx, g.model, genai.Text(prompt), config)
			return stream, nil
		})
}

func (g *GeminiChatClient) Model() string {
	return g.model
}

func (g *GeminiChatClient) Close() error {
	if g == nil || g.client == nil {
		return nil
	}
	return nil
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
	return ExtractText(result)
}

func (cs *ChatSession) SendMessageStream(ctx context.Context, message string) iter.Seq2[*genai.GenerateContentResponse, error] {
	return cs.chat.SendMessageStream(ctx, genai.Part{Text: message})
}