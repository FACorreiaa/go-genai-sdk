package genai_sdk

import (
	"regexp"
	"strings"
)

// sectionTagPattern matches a leading streaming section marker such as
// "[nearby_pois]\n" that consolidation steps prepend to each response part.
var sectionTagPattern = regexp.MustCompile(`^\s*\[[a-z_]+\]\s*`)

// CleanJSON normalizes LLM JSON output: strips section tags, markdown fences,
// bare null/empty sentinels, and extracts the first balanced JSON object.
func CleanJSON(raw string) string {
	response := strings.TrimSpace(raw)

	response = sectionTagPattern.ReplaceAllString(response, "")
	response = strings.TrimSpace(response)

	switch response {
	case "", "null", "[]", "{}":
		return ""
	}

	codeBlockPattern := regexp.MustCompile("(?s)```(?:json)?\\s*([\\s\\S]*?)```")
	if matches := codeBlockPattern.FindStringSubmatch(response); len(matches) > 1 {
		response = strings.TrimSpace(matches[1])
	} else {
		if after, ok := strings.CutPrefix(response, "```json"); ok {
			response = after
		} else if after, ok := strings.CutPrefix(response, "```"); ok {
			response = after
		}
		response = strings.TrimSuffix(response, "```")
		response = strings.TrimSpace(response)
	}

	firstBrace := strings.Index(response, "{")
	if firstBrace == -1 {
		return response
	}

	braceCount := 0
	lastValidBrace := -1
	inString := false
	escapeNext := false

	for i := firstBrace; i < len(response); i++ {
		char := response[i]

		if escapeNext {
			escapeNext = false
			continue
		}
		if char == '\\' {
			escapeNext = true
			continue
		}
		if char == '"' {
			inString = !inString
			continue
		}

		if !inString {
			switch char {
			case '{':
				braceCount++
			case '}':
				braceCount--
				if braceCount == 0 {
					lastValidBrace = i
					break
				}
			}
		}
	}

	if braceCount != 0 {
		lastBrace := strings.LastIndex(response, "}")
		if lastBrace == -1 || lastBrace <= firstBrace {
			return response
		}
		lastValidBrace = lastBrace
	}

	if lastValidBrace == -1 {
		return response
	}

	jsonPortion := response[firstBrace : lastValidBrace+1]
	jsonPortion = strings.ReplaceAll(jsonPortion, "`", "")
	jsonPortion = regexp.MustCompile(`,(\s*[}\\]])`).ReplaceAllString(jsonPortion, "$1")

	return strings.TrimSpace(jsonPortion)
}

// IsEmptyJSON reports whether raw LLM output carries no parseable JSON payload.
func IsEmptyJSON(raw string) bool {
	return CleanJSON(raw) == ""
}