package genai_sdk

import (
	"errors"
	"fmt"
	"iter"
	"strings"

	"google.golang.org/genai"
)

// ErrNoContent reports that a response carried nothing of the kind asked for.
//
// Worth a sentinel rather than a message because it is not always a failure: a
// recording of silence transcribes to no text, and a caller that can tell that
// apart from a broken call answers "I could not hear anything" instead of
// "something went wrong".
var ErrNoContent = errors.New("no content in response")

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

	return "", fmt.Errorf("no text content in response: %w", ErrNoContent)
}

// ExtractAudio returns the first audio part of a response.
//
// The failure worth naming is a model that answered with words instead of
// sound: that is what a model which does not support AUDIO output does, rather
// than refusing the request, and it is the shape of every misconfigured speech
// model. Saying so here saves reading it as an empty-response bug.
func ExtractAudio(resp *genai.GenerateContentResponse) (Audio, error) {
	if resp == nil {
		return Audio{}, fmt.Errorf("response is nil")
	}

	var spokeInstead bool
	for _, cand := range resp.Candidates {
		if cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "audio/") {
				return Audio{
					Data:     part.InlineData.Data,
					MIMEType: part.InlineData.MIMEType,
				}, nil
			}
			if part.Text != "" {
				spokeInstead = true
			}
		}
	}

	if spokeInstead {
		return Audio{}, fmt.Errorf("the model answered with text rather than audio; it is probably not a speech model: %w", ErrNoContent)
	}
	return Audio{}, fmt.Errorf("no audio content in response: %w", ErrNoContent)
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
