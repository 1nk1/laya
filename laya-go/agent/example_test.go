package agent_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"testing/fstest"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// scorer is a stand-in for a native session: it prefers the second option of every question.
type scorer struct{}

func (scorer) Run(_ context.Context, b infer.Batch) (infer.Output, error) {
	out := infer.Output{Logits: make([]float32, b.Rows*b.Markers), ActLogits: make([]float32, b.Rows*2)}
	for i := 0; i < b.Rows; i++ {
		out.Logits[i*b.Markers+1] = 2
	}
	return out, nil
}

func (scorer) Close() error { return nil }

func Example() {
	// A checkpoint is a directory with rl_agent_config.json and tokenizer/. This one is a toy.
	spec, cfg := fixtures.ToyTokenizer()
	checkpoint := fstest.MapFS{
		"rl_agent_config.json":            {Data: []byte(`{"max_len": 128, "head_max_len": 64}`)},
		"tokenizer/tokenizer.json":        {Data: spec},
		"tokenizer/tokenizer_config.json": {Data: cfg},
	}
	a, err := agent.Open(checkpoint, scorer{}) // a real session comes from the onnx package
	if err != nil {
		log.Fatal(err)
	}
	defer a.Close()

	qs, err := schema.NewQuestions(
		schema.Ask("intent", schema.MustQuestion(schema.ChoiceOf("What does the customer want?", "status", "refund"))),
		schema.Ask("angry", schema.MustQuestion(schema.Noul("Is the customer angry?"))),
	)
	if err != nil {
		log.Fatal(err)
	}
	preds, err := a.PredictBatch(context.Background(), []any{"I was charged twice", "Where is my order?"}, qs,
		agent.WithBatchSize(2), agent.WithSortByLength())
	if err != nil {
		log.Fatal(err)
	}
	for _, p := range preds {
		intent, _ := p.Answers.Get("intent")
		fmt.Println(intent.(*schema.ChoiceAnswer).Choice, p.Usage.InputTokens, p.Usage.Truncated)
	}
	fmt.Println(a.Config().MaxLen)
	// Output:
	// refund 170 false
	// refund 168 false
	// 128
}

// ExampleOpen_checkpoint is compiled and not run: it needs a checkpoint and a native session.
func ExampleOpen_checkpoint() {
	var session infer.Session // from the onnx package
	a, err := agent.Open(os.DirFS("checkpoints/english"), session)
	if err != nil {
		log.Fatal(err)
	}
	defer a.Close()
	qs := schema.Questions{}
	if _, err := a.Predict(context.Background(), "I was charged twice", qs, agent.WithLanguage("en")); err != nil {
		log.Fatal(err)
	}
}
