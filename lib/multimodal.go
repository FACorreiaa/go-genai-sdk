package genai_sdk

import (
	"context"
	"fmt"

	"google.golang.org/genai"
)

// Blob is one binary input to a model: the bytes, and the MIME type that tells
// the model what they are.
type Blob struct {
	Data     []byte
	MIMEType string
}

// MultimodalClient generates from a prompt plus binary parts — audio to
// transcribe, an image to describe.
//
// Separate from ChatClient rather than added to it. Every consumer wraps
// ChatClient several times over — metering, tracing, fallback chains, per-user
// keys — and a method added there is a compile break in all of them for a
// capability only one implementation has.
type MultimodalClient interface {
	GenerateFromParts(ctx context.Context, prompt string, blobs []Blob, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error)
	GenerateTextFromParts(ctx context.Context, prompt string, blobs []Blob, config *genai.GenerateContentConfig) (string, error)
}

// GenerateFromParts sends the prompt and the blobs as one turn.
//
// The prompt goes first. Gemini reads a trailing instruction as commentary on
// the media rather than as the task, and "transcribe this" after a minute of
// audio is noticeably less reliable than the same words before it.
func (g *GeminiChatClient) GenerateFromParts(ctx context.Context, prompt string, blobs []Blob, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
	if g == nil || g.client == nil {
		return nil, fmt.Errorf("client not initialized")
	}
	if len(blobs) == 0 {
		return nil, fmt.Errorf("no binary parts to generate from")
	}

	parts := make([]*genai.Part, 0, len(blobs)+1)
	if prompt != "" {
		parts = append(parts, genai.NewPartFromText(prompt))
	}
	for _, blob := range blobs {
		if len(blob.Data) == 0 {
			return nil, fmt.Errorf("a binary part carries no data")
		}
		if blob.MIMEType == "" {
			return nil, fmt.Errorf("a binary part carries no MIME type")
		}
		parts = append(parts, genai.NewPartFromBytes(blob.Data, blob.MIMEType))
	}
	contents := []*genai.Content{genai.NewContentFromParts(parts, genai.RoleUser)}

	return retryWithBackoff(ctx, g.retryPolicy, g.logger, "GenerateFromParts",
		func() (*genai.GenerateContentResponse, error) {
			return g.client.Models.GenerateContent(ctx, g.model, contents, config)
		})
}

// GenerateTextFromParts returns just the text of a multimodal generation.
func (g *GeminiChatClient) GenerateTextFromParts(ctx context.Context, prompt string, blobs []Blob, config *genai.GenerateContentConfig) (string, error) {
	resp, err := g.GenerateFromParts(ctx, prompt, blobs, config)
	if err != nil {
		return "", err
	}
	return ExtractText(resp)
}

var _ MultimodalClient = (*GeminiChatClient)(nil)
