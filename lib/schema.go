package genai_sdk

import "google.golang.org/genai"

// JSONModeConfig returns a GenerateContentConfig that requests JSON matching schema.
func JSONModeConfig(schema *genai.Schema, temperature float32) *genai.GenerateContentConfig {
	return &genai.GenerateContentConfig{
		Temperature:      genai.Ptr(temperature),
		ResponseMIMEType: "application/json",
		ResponseSchema:   schema,
	}
}

// POIListSchema describes the points_of_interest JSON contract used by Loci POI prompts.
func POIListSchema() *genai.Schema {
	return &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"points_of_interest": {
				Type: genai.TypeArray,
				Items: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"name":            {Type: genai.TypeString},
						"latitude":        {Type: genai.TypeNumber},
						"longitude":       {Type: genai.TypeNumber},
						"category":        {Type: genai.TypeString},
						"description_poi": {Type: genai.TypeString},
					},
					Required: []string{"name", "latitude", "longitude", "category", "description_poi"},
				},
			},
		},
		Required: []string{"points_of_interest"},
	}
}