package genai_sdk

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"google.golang.org/genai"
)

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"context canceled", context.Canceled, false},
		{"context deadline", context.DeadlineExceeded, false},
		{"api 429", genai.APIError{Code: 429}, true},
		{"api 503", genai.APIError{Code: 503}, true},
		{"api 500", genai.APIError{Code: 500}, true},
		{"api 400 not retryable", genai.APIError{Code: 400}, false},
		{"api 404 not retryable", genai.APIError{Code: 404}, false},
		{"wrapped api 503", fmt.Errorf("call failed: %w", genai.APIError{Code: 503}), true},
		{"connection reset string", errors.New("read: connection reset by peer"), true},
		{"timeout string", errors.New("context deadline: i/o timeout"), true},
		{"plain error", errors.New("some logic error"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isRetryable(tt.err); got != tt.want {
				t.Errorf("isRetryable(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

// fastPolicy keeps backoff tiny so tests run quickly.
var fastPolicy = RetryPolicy{MaxRetries: 3, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond}

func TestRetryWithBackoff_SuccessAfterRetry(t *testing.T) {
	calls := 0
	got, err := retryWithBackoff(context.Background(), fastPolicy, nil, "test",
		func() (string, error) {
			calls++
			if calls < 3 {
				return "", genai.APIError{Code: 503}
			}
			return "ok", nil
		})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "ok" {
		t.Errorf("got %q, want ok", got)
	}
	if calls != 3 {
		t.Errorf("expected 3 calls, got %d", calls)
	}
}

func TestRetryWithBackoff_ExhaustsRetries(t *testing.T) {
	calls := 0
	_, err := retryWithBackoff(context.Background(), fastPolicy, nil, "test",
		func() (string, error) {
			calls++
			return "", genai.APIError{Code: 503}
		})
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	// MaxRetries=3 means 1 initial + 3 retries = 4 attempts.
	if calls != 4 {
		t.Errorf("expected 4 calls, got %d", calls)
	}
}

func TestRetryWithBackoff_NonRetryablePassthrough(t *testing.T) {
	calls := 0
	_, err := retryWithBackoff(context.Background(), fastPolicy, nil, "test",
		func() (string, error) {
			calls++
			return "", genai.APIError{Code: 400}
		})
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Errorf("non-retryable error should not retry; got %d calls", calls)
	}
}

func TestRetryWithBackoff_ContextCancelDuringBackoff(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	// Cancel after the first failure so the backoff wait aborts.
	_, err := retryWithBackoff(ctx, RetryPolicy{MaxRetries: 3, BaseDelay: time.Second, MaxDelay: time.Second}, nil, "test",
		func() (string, error) {
			calls++
			cancel()
			return "", genai.APIError{Code: 503}
		})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
	if calls != 1 {
		t.Errorf("expected 1 call before cancel abort, got %d", calls)
	}
}

func TestRetryWithBackoff_ImmediateSuccess(t *testing.T) {
	calls := 0
	got, err := retryWithBackoff(context.Background(), fastPolicy, nil, "test",
		func() (int, error) {
			calls++
			return 42, nil
		})
	if err != nil || got != 42 || calls != 1 {
		t.Errorf("got=%d err=%v calls=%d; want 42, nil, 1", got, err, calls)
	}
}
