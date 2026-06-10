package genai_sdk

import "testing"

func TestCleanJSON(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"bare null", "null", ""},
		{"section tagged null", "[nearby_pois]\nnull", ""},
		{"section tagged empty array", "[nearby_pois]\n[]", ""},
		{"empty array", "[]", ""},
		{"empty object", "{}", ""},
		{"whitespace", "   \n  ", ""},
		{"section tag with json object",
			"[nearby_pois]\n{\"points_of_interest\": [{\"name\": \"X\"}]}",
			"{\"points_of_interest\": [{\"name\": \"X\"}]}"},
		{"plain json passthrough",
			"{\"points_of_interest\": []}",
			"{\"points_of_interest\": []}"},
		{"markdown fenced json",
			"```json\n{\"a\": 1}\n```",
			"{\"a\": 1}"},
		{"section tag then fenced json",
			"[nearby_pois]\n```json\n{\"a\": 1}\n```",
			"{\"a\": 1}"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CleanJSON(tt.in); got != tt.want {
				t.Errorf("CleanJSON(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestIsEmptyJSON(t *testing.T) {
	if !IsEmptyJSON("null") {
		t.Error("expected null to be empty")
	}
	if IsEmptyJSON(`{"a":1}`) {
		t.Error("expected object to be non-empty")
	}
}