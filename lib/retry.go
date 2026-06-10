package genai_sdk

import (
	"context"
	"errors"
	"log/slog"
	"math/rand"
	"strings"
	"time"

	"google.golang.org/genai"
)

// RetryPolicy controls how transient LLM call failures are retried.
type RetryPolicy struct {
	// MaxRetries is the number of retries after the initial attempt.
	// MaxRetries=3 means up to 4 total attempts.
	MaxRetries int
	// BaseDelay is the delay before the first retry; it grows exponentially.
	BaseDelay time.Duration
	// MaxDelay caps the per-attempt backoff delay.
	MaxDelay time.Duration
}

// DefaultRetryPolicy is a sane default for chat/content generation calls.
var DefaultRetryPolicy = RetryPolicy{
	MaxRetries: 3,
	BaseDelay:  500 * time.Millisecond,
	MaxDelay:   8 * time.Second,
}

// retryableStatusCodes are HTTP status codes worth retrying with backoff.
var retryableStatusCodes = map[int]bool{
	429: true,
	500: true,
	502: true,
	503: true,
	504: true,
}

// IsRetryable reports whether an error from the genai client is transient and
// worth retrying. Context cancellation/deadline are intentionally NOT retried.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}

	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return retryableStatusCodes[apiErr.Code]
	}

	msg := strings.ToLower(err.Error())
	for _, frag := range []string{
		"connection reset",
		"connection refused",
		"timeout",
		"timed out",
		"temporarily unavailable",
		"eof",
		"too many requests",
		"unavailable",
		"resource_exhausted",
		"quota",
	} {
		if strings.Contains(msg, frag) {
			return true
		}
	}
	return false
}

func backoffDelay(policy RetryPolicy, attempt int) time.Duration {
	delay := policy.BaseDelay << attempt
	if delay <= 0 || delay > policy.MaxDelay {
		delay = policy.MaxDelay
	}
	half := delay / 2
	return half + time.Duration(rand.Int63n(int64(half)+1))
}

func retryWithBackoff[T any](
	ctx context.Context,
	policy RetryPolicy,
	logger *slog.Logger,
	op string,
	fn func() (T, error),
) (T, error) {
	var result T
	var err error

	for attempt := 0; attempt <= policy.MaxRetries; attempt++ {
		result, err = fn()
		if err == nil || !IsRetryable(err) || attempt == policy.MaxRetries {
			return result, err
		}

		delay := backoffDelay(policy, attempt)
		if logger != nil {
			logger.WarnContext(ctx, "retrying LLM call after transient error",
				slog.String("op", op),
				slog.Int("attempt", attempt+1),
				slog.Int("max_retries", policy.MaxRetries),
				slog.Duration("delay", delay),
				slog.String("error", err.Error()),
			)
		}

		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return result, ctx.Err()
		case <-timer.C:
		}
	}
	return result, err
}