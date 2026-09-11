package genai_sdk

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/genai"
)

// fakeGemini serves one canned generateContent response and records the last
// request body, so a test can assert on what was actually sent.
type fakeGemini struct {
	server *httptest.Server
	body   string
	path   string
}

func newFakeGemini(t *testing.T, response string) *fakeGemini {
	t.Helper()
	f := &fakeGemini{}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		f.body = string(raw)
		f.path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(response))
	}))
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeGemini) client(t *testing.T, model string) *GeminiChatClient {
	t.Helper()
	c, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:      "test-key",
		Backend:     genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: f.server.URL},
	})
	if err != nil {
		t.Fatalf("could not build a client against the test server: %v", err)
	}
	return &GeminiChatClient{client: c, model: model, retryPolicy: RetryPolicy{}}
}

func audioResponse(mimeType string, data []byte) string {
	return fmt.Sprintf(`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":%q,"data":%q}}]}}]}`,
		mimeType, base64.StdEncoding.EncodeToString(data))
}

func TestGenerateSpeechReturnsPCM(t *testing.T) {
	want := []byte{0x01, 0x02, 0x03, 0x04}
	fake := newFakeGemini(t, audioResponse("audio/L16;codec=pcm;rate=24000", want))
	client := fake.client(t, "gemini-2.5-flash")

	got, err := client.GenerateSpeech(context.Background(), "a weekend in Porto", SpeechOptions{
		Model:     "gemini-2.5-flash-preview-tts",
		VoiceName: "Kore",
	})
	if err != nil {
		t.Fatalf("GenerateSpeech: %v", err)
	}
	if string(got.Data) != string(want) {
		t.Errorf("audio data = %v, want %v", got.Data, want)
	}
	if got.MIMEType != "audio/L16;codec=pcm;rate=24000" {
		t.Errorf("mime type = %q", got.MIMEType)
	}

	// The voice and the AUDIO modality have to survive the trip, or the model
	// answers with text and the failure looks like an empty response.
	if !strings.Contains(fake.body, `"Kore"`) {
		t.Errorf("request did not carry the voice name: %s", fake.body)
	}
	if !strings.Contains(fake.body, "AUDIO") {
		t.Errorf("request did not ask for the AUDIO modality: %s", fake.body)
	}
}

func TestGenerateSpeechUsesOptionModelOverClientModel(t *testing.T) {
	fake := newFakeGemini(t, audioResponse("audio/L16;rate=24000", []byte{0x00}))
	client := fake.client(t, "gemini-2.5-flash")

	if _, err := client.GenerateSpeech(context.Background(), "hello", SpeechOptions{
		Model: "gemini-2.5-flash-preview-tts",
	}); err != nil {
		t.Fatalf("GenerateSpeech: %v", err)
	}

	// The model is in the URL path, and a chat model cannot synthesise: if the
	// override is dropped the request silently goes to the wrong model and
	// comes back as text.
	if !strings.Contains(fake.path, "gemini-2.5-flash-preview-tts") {
		t.Errorf("request went to %q, want the TTS model", fake.path)
	}
}

func TestGenerateSpeechRejectsEmptyText(t *testing.T) {
	fake := newFakeGemini(t, audioResponse("audio/L16", []byte{0x00}))
	client := fake.client(t, "gemini-2.5-flash")

	if _, err := client.GenerateSpeech(context.Background(), "", SpeechOptions{}); err == nil {
		t.Error("expected an error for empty text")
	}
}

func TestExtractAudioNamesAModelThatAnsweredWithText(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Parts: []*genai.Part{{Text: "I cannot produce audio."}}},
		}},
	}

	_, err := ExtractAudio(resp)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(err, ErrNoContent) {
		t.Errorf("error should wrap ErrNoContent, got %v", err)
	}
	// This is the message somebody reads at 2am after pointing the TTS model
	// at a chat model; it has to say so.
	if !strings.Contains(err.Error(), "not a speech model") {
		t.Errorf("error should name the likely cause, got %q", err)
	}
}

func TestExtractAudioSkipsNonAudioInlineData(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{0x89}}},
				{InlineData: &genai.Blob{MIMEType: "audio/L16;rate=24000", Data: []byte{0x42}}},
			}},
		}},
	}

	got, err := ExtractAudio(resp)
	if err != nil {
		t.Fatalf("ExtractAudio: %v", err)
	}
	if len(got.Data) != 1 || got.Data[0] != 0x42 {
		t.Errorf("picked the wrong part: %v", got)
	}
}
