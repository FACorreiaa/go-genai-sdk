package genai_sdk

import (
	"context"
	"encoding/base64"
	"errors"
	"strings"
	"testing"
)

func textResponse(text string) string {
	return `{"candidates":[{"content":{"role":"model","parts":[{"text":"` + text + `"}]}}]}`
}

func TestGenerateTextFromPartsSendsAudioInline(t *testing.T) {
	fake := newFakeGemini(t, textResponse("three days in Lisbon"))
	client := fake.client(t, "gemini-2.5-flash")

	audio := []byte{0xDE, 0xAD, 0xBE, 0xEF}
	got, err := client.GenerateTextFromParts(context.Background(), "Transcribe this.",
		[]Blob{{Data: audio, MIMEType: "audio/ogg"}}, nil)
	if err != nil {
		t.Fatalf("GenerateTextFromParts: %v", err)
	}
	if got != "three days in Lisbon" {
		t.Errorf("transcript = %q", got)
	}

	if !strings.Contains(fake.body, base64.StdEncoding.EncodeToString(audio)) {
		t.Errorf("request did not carry the audio bytes: %s", fake.body)
	}
	if !strings.Contains(fake.body, "audio/ogg") {
		t.Errorf("request did not carry the MIME type: %s", fake.body)
	}

	// The instruction has to precede the media: Gemini reads a trailing
	// instruction as commentary on the recording rather than as the task.
	promptAt := strings.Index(fake.body, "Transcribe this.")
	audioAt := strings.Index(fake.body, "audio/ogg")
	if promptAt < 0 || audioAt < 0 || promptAt > audioAt {
		t.Errorf("prompt should come before the audio part: %s", fake.body)
	}
}

func TestGenerateFromPartsRejectsBadInput(t *testing.T) {
	fake := newFakeGemini(t, textResponse("unused"))
	client := fake.client(t, "gemini-2.5-flash")

	tests := []struct {
		name  string
		blobs []Blob
	}{
		{"no parts", nil},
		{"no data", []Blob{{MIMEType: "audio/ogg"}}},
		{"no mime type", []Blob{{Data: []byte{0x01}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := client.GenerateFromParts(context.Background(), "x", tt.blobs, nil); err == nil {
				t.Error("expected an error")
			}
		})
	}
}

func TestEmptyTranscriptIsDistinguishable(t *testing.T) {
	// A recording of silence comes back with no text. The caller has to be
	// able to tell that apart from a broken call, or it answers "something
	// went wrong" when the honest answer is "I could not hear anything".
	fake := newFakeGemini(t, `{"candidates":[{"content":{"role":"model","parts":[]}}]}`)
	client := fake.client(t, "gemini-2.5-flash")

	_, err := client.GenerateTextFromParts(context.Background(), "Transcribe this.",
		[]Blob{{Data: []byte{0x01}, MIMEType: "audio/ogg"}}, nil)
	if !errors.Is(err, ErrNoContent) {
		t.Errorf("error should wrap ErrNoContent, got %v", err)
	}
}
