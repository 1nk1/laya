package infer_test

import (
	"errors"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/infer"
	"github.com/NandhaKishorM/laya/laya-go/internal/collate"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func batch(rows, length, markers int) infer.Batch {
	return infer.Batch{
		InputIDs: make([]int64, rows*length), AttentionMask: make([]int64, rows*length),
		MarkerPos: make([]int64, rows*markers), MarkerMask: make([]bool, rows*markers),
		QType: make([]int64, rows), Rows: rows, Length: length, Markers: markers,
	}
}

func TestBatchValidate(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*infer.Batch)
		field  string // the ShapeError's tensor, "" for none, "other" for a non-ShapeError
	}{
		{"consistent", func(*infer.Batch) {}, ""},
		{"empty", func(b *infer.Batch) { *b = infer.Batch{} }, ""},
		{"short input_ids", func(b *infer.Batch) { b.InputIDs = b.InputIDs[1:] }, "input_ids"},
		{"long attention_mask", func(b *infer.Batch) { b.AttentionMask = append(b.AttentionMask, 0) }, "attention_mask"},
		{"short marker_pos", func(b *infer.Batch) { b.MarkerPos = nil }, "marker_pos"},
		{"short marker_mask", func(b *infer.Batch) { b.MarkerMask = b.MarkerMask[:1] }, "marker_mask"},
		{"short qtype", func(b *infer.Batch) { b.QType = nil }, "qtype"},
		{"one marker slot", func(b *infer.Batch) { *b = batch(3, 4, 1) }, "other"},
		{"negative rows", func(b *infer.Batch) { b.Rows = -1 }, "other"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b := batch(3, 4, 2)
			c.mutate(&b)
			err := b.Validate()
			var se *infer.ShapeError
			switch {
			case c.field == "" && err != nil:
				t.Fatalf("Validate = %v, want nil", err)
			case c.field == "other" && (err == nil || errors.As(err, &se)):
				t.Fatalf("Validate = %v, want a plain error", err)
			case c.field != "" && c.field != "other" && (!errors.As(err, &se) || se.Name != c.field):
				t.Fatalf("Validate = %v, want a ShapeError for %s", err, c.field)
			}
		})
	}
}

func TestOutputCheck(t *testing.T) {
	b := batch(3, 4, 5)
	good := infer.Output{Logits: make([]float32, 15), ActLogits: make([]float32, 6)}
	if err := good.Check(b); err != nil {
		t.Fatalf("Check = %v", err)
	}
	for name, o := range map[string]infer.Output{
		"logits":     {Logits: make([]float32, 14), ActLogits: make([]float32, 6)},
		"act_logits": {Logits: make([]float32, 15), ActLogits: make([]float32, 5)},
	} {
		var se *infer.ShapeError
		if err := o.Check(b); !errors.As(err, &se) || se.Name != name {
			t.Errorf("%s: Check = %v", name, err)
		} else if se.Got == se.Want || se.Error() == "" {
			t.Errorf("%s: %+v", name, se)
		}
	}
}

// TestCollateBatchIsInferBatch pins the alias: a Batch the collator builds is a valid infer.Batch.
func TestCollateBatchIsInferBatch(t *testing.T) {
	b, err := collate.Collate([]collate.Row{
		{IDs: []int32{1, 2, 3}, Markers: []int{1}, Type: schema.TypeChoice},
		{IDs: []int32{4, 5}, Markers: []int{0, 1}, Type: schema.TypeNoul},
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	var ib infer.Batch = b
	if err := ib.Validate(); err != nil || ib.Rows != 2 || ib.Length != 3 || ib.Markers != 2 {
		t.Fatalf("Validate = %v on %+v", err, ib)
	}
}
