package genai_sdk

import (
	"context"
	"fmt"

	"google.golang.org/genai"
)

// SpeechOptions selects what a synthesis request sounds like. Zero values mean
// the provider's default.
type SpeechOptions struct {
	// Model overrides the client's model, and is required in practice:
	// Gemini's speech models are distinct from its chat models, so a client
	// built for chat cannot synthesise without it.
	Model string
	// VoiceName is one of Gemini's prebuilt voices — "Kore", "Puck", "Aoede".
	VoiceName string
	// LanguageCode is ISO 639-1. Empty lets the model follow the text, which
	// is usually what you want when the text came from a user.
	LanguageCode string
}

// Audio is synthesised speech: the bytes, and what they are.
//
// Gemini answers with raw little-endian 16-bit PCM at 24 kHz, described as
// "audio/L16;codec=pcm;rate=24000". That is not a container format — anything
// that has to play it, or hand it to something that plays it, has to wrap or
// transcode it first.
type Audio struct {
	Data     []byte
	MIMEType string
}

// SpeechClient turns text into speech.
//
// Separate from ChatClient for the same reason MultimodalClient is: the
// wrappers around ChatClient are many and none of them can synthesise.
type SpeechClient interface {
	GenerateSpeech(ctx context.Context, text string, opts SpeechOptions) (Audio, error)
}

// GenerateSpeech renders text as audio.
func (g *GeminiChatClient) GenerateSpeech(ctx context.Context, text string, opts SpeechOptions) (Audio, error) {
	if g == nil || g.client == nil {
		return Audio{}, fmt.Errorf("client not initialized")
	}
	if text == "" {
		return Audio{}, fmt.Errorf("nothing to say")
	}

	model := opts.Model
	if model == "" {
		model = g.model
	}

	// A fresh config rather than one the caller passes in: every other field
	// on GenerateContentConfig is about generating text, and a JSON schema or
	// a response MIME type left over from a chat call makes this request fail
	// in a way that reads as a model problem.
	config := &genai.GenerateContentConfig{
		ResponseModalities: []string{"AUDIO"},
		SpeechConfig: &genai.SpeechConfig{
			LanguageCode: opts.LanguageCode,
		},
	}
	if opts.VoiceName != "" {
		config.SpeechConfig.VoiceConfig = &genai.VoiceConfig{
			PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{VoiceName: opts.VoiceName},
		}
	}

	resp, err := retryWithBackoff(ctx, g.retryPolicy, g.logger, "GenerateSpeech",
		func() (*genai.GenerateContentResponse, error) {
			return g.client.Models.GenerateContent(ctx, model, genai.Text(text), config)
		})
	if err != nil {
		return Audio{}, err
	}
	return ExtractAudio(resp)
}

var _ SpeechClient = (*GeminiChatClient)(nil)
