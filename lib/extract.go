package genai_sdk

import (
	"fmt"
	"iter"
	"strings"

	"google.golang.org/genai"
)

// ExtractText returns concatenated text from the first candidate with content.
func ExtractText(resp *genai.GenerateContentResponse) (string, error) {
	if resp == nil {
		return "", fmt.Errorf("response is nil")
	}

	var b strings.Builder
	for _, cand := range resp.Candidates {
		if cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if part.Text != "" {
				b.WriteString(part.Text)
			}
		}
		if b.Len() > 0 {
			return b.String(), nil
		}
	}

	return "", fmt.Errorf("no text content in response")
}

// ExtractUsage returns prompt, completion, and total token counts when present.
func ExtractUsage(resp *genai.GenerateContentResponse) (prompt, completion, total int) {
	if resp == nil || resp.UsageMetadata == nil {
		return 0, 0, 0
	}
	um := resp.UsageMetadata
	return int(um.PromptTokenCount), int(um.CandidatesTokenCount), int(um.TotalTokenCount)
}

// ConcatStreamText drains a generate-content stream into a single string.
func ConcatStreamText(stream iter.Seq2[*genai.GenerateContentResponse, error]) (string, error) {
	var b strings.Builder
	for resp, err := range stream {
		if err != nil {
			return b.String(), err
		}
		if resp == nil {
			continue
		}
		b.WriteString(resp.Text())
	}
	return b.String(), nil
}