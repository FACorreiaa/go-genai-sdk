package genai_sdk

import (
	"testing"

	"google.golang.org/genai"
)

func TestExtractText(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{
				Parts: []*genai.Part{{Text: "hello world"}},
			},
		}},
	}
	got, err := ExtractText(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "hello world" {
		t.Errorf("got %q, want hello world", got)
	}
}

func TestExtractText_Empty(t *testing.T) {
	_, err := ExtractText(&genai.GenerateContentResponse{})
	if err == nil {
		t.Fatal("expected error for empty response")
	}
}

func TestExtractUsage(t *testing.T) {
	resp := &genai.GenerateContentResponse{
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     10,
			CandidatesTokenCount: 20,
			TotalTokenCount:      30,
		},
	}
	p, c, tot := ExtractUsage(resp)
	if p != 10 || c != 20 || tot != 30 {
		t.Errorf("got (%d,%d,%d), want (10,20,30)", p, c, tot)
	}
}