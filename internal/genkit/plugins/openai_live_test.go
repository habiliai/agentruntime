package plugins

import (
	"context"
	"encoding/json"
	"flag"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/genkit"

	"github.com/firebase/genkit/go/ai"
	oai "github.com/firebase/genkit/go/plugins/compat_oai/openai"
)

// The tests here only work with an API key set to a valid value.
var apiKey = flag.String("key", "", "OpenAI API key")

// We can't test the DefineAll functions along with the other tests because
// we get duplicate definitions of models.
var testAll = flag.Bool("all", false, "test DefineAllXXX functions")

func TestLive(t *testing.T) {
	if *apiKey == "" {
		t.Skipf("no -key provided")
	}
	if *testAll {
		t.Skip("-all provided")
	}
	ctx := context.Background()
	g, err := genkit.Init(ctx, genkit.WithPlugins(&oai.OpenAI{
		APIKey: *apiKey,
	}))
	if err != nil {
		t.Fatal(err)
	}

	embedder := (&oai.OpenAI{}).Embedder(g, "text-embedding-3-small")
	model := (&oai.OpenAI{}).Model(g, "gpt-4o-mini")

	t.Run("embedder", func(t *testing.T) {
		res, err := ai.Embed(ctx, embedder, ai.WithTextDocs("yellow banana"))
		if err != nil {
			t.Fatal(err)
		}
		out := res.Embeddings[0].Embedding
		// There's not a whole lot we can test about the result.
		// Just do a few sanity checks.
		if len(out) < 100 {
			t.Errorf("embedding vector looks too short: len(out)=%d", len(out))
		}
		var normSquared float32
		for _, x := range out {
			normSquared += x * x
		}
		if normSquared < 0.9 || normSquared > 1.1 {
			t.Errorf("embedding vector not unit length: %f", normSquared)
		}
	})

	t.Run("generate", func(t *testing.T) {
		resp, err := genkit.Generate(
			ctx, //
			g,
			ai.WithModel(model),
			ai.WithPrompt("Just the country name where Napoleon was emperor, no period."), //
			ai.WithConfig(map[string]interface{}{
				"temperature": 1.0,
			}),
		)
		if err != nil {
			t.Fatal(err)
		}
		out := resp.Text()
		const want = "France"
		if out != want {
			t.Errorf("got %q, expecting %q", out, want)
		}
		if resp.Request == nil {
			t.Error("Request field not set properly")
		}
		if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 || resp.Usage.TotalTokens == 0 {
			t.Errorf("Empty usage stats %#v", *resp.Usage)
		}
	})

	t.Run("tool", func(t *testing.T) {
		gablorkenTool := genkit.DefineTool(g, "gablorken", "use when need to calculate a gablorken",
			func(
				ctx *ai.ToolContext,
				input struct {
					Value float64
					Over  float64
				},
			) (float64, error) {
				return math.Pow(input.Value, input.Over), nil
			},
		)
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithModel(model), //
			ai.WithPrompt("What is a gablorken of 2 over 3.5?"), //
			ai.WithTools(gablorkenTool),                         //
		)
		if err != nil {
			t.Fatal(err)
		}
		out := resp.Text()
		const want = "12.25"
		if !strings.Contains(out, want) {
			t.Errorf("got %q, expecting it to contain %q", out, want)
		}
	})

	t.Run("structured output", func(t *testing.T) {
		type User struct {
			Name string
			Age  int
		}
		resp, err := genkit.Generate(
			ctx, //
			g,
			ai.WithModel(model), //
			ai.WithPrompt("Create dummy user data with the name John and age 32."), //
			ai.WithOutputType(User{}), //
		)
		if err != nil {
			t.Fatal(err)
		}
		respText := resp.Text()

		out := &User{}
		if err := json.Unmarshal([]byte(respText), out); err != nil {
			t.Fatal(err)
		}

		want := &User{Name: "John", Age: 32}
		if !reflect.DeepEqual(out, want) {
			t.Errorf("got %q, expecting %q", out, want)
		}
	})
}
