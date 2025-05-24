package xai

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
)

// The tests here only work with an API key set to a valid value.
var apiKey = flag.String("key", "", "XAI API key")

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
	g, err := genkit.Init(ctx, genkit.WithPlugins(&Plugin{
		APIKey: *apiKey,
	}))
	if err != nil {
		t.Fatal(err)
	}

	model := Model(g, "grok-3-mini")

	t.Run("generate", func(t *testing.T) {
		resp, err := genkit.Generate(
			ctx, //
			g,
			ai.WithModel(model),
			ai.WithPrompt("Just the country name where Napoleon was emperor, no period."), //
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
		gablorkenTool := genkit.DefineTool(g, "gablorken", "use when need to calculate a gablorken (power/exponent calculation)",
			func(
				ctx *ai.ToolContext,
				input struct {
				Value float64
				Over  float64
			},
			) (float64, error) {
				output := math.Pow(input.Value, input.Over)
				return output, nil
			},
		)
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithModel(model),                                 //
			ai.WithPrompt("What is a gablorken of 2 over 3.5? Use the gablorken tool to calculate this."), //
			ai.WithTools(gablorkenTool),                         //
		)
		if err != nil {
			t.Fatal(err)
		}
		out := resp.Text()
		const want = "11"
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
			ai.WithModel(model),                                                    //
			ai.WithPrompt("Create dummy user data with the name John and age 32."), //
			ai.WithOutputType(User{}),                                              //
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
