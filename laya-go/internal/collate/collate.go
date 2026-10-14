// Package collate pads built sequences into the rectangular tensors the graph takes. The
// tensor layout is the ONNX graph's contract, not a caller's, so this stays internal.
//
// One row per question, not per state: four questions about one document are four rows that
// share a state. Rows are padded to the longest sequence and markers to the most options.
package collate

import (
	"errors"
	"fmt"

	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// MinMarkerSlots is the fewest option slots the graph accepts: the exported head takes
// topk(2), so one-option rows get a masked second slot that decoding never reads.
const MinMarkerSlots = 2

// ErrEmpty is returned for a batch with no rows.
var ErrEmpty = errors.New("nothing to collate")

// Row is one sequence to collate.
type Row struct {
	IDs     []int32
	Markers []int
	Type    schema.Type
}

// Batch is the graph's five inputs, each flattened row-major, as onnxruntime tensors take them.
type Batch struct {
	InputIDs              []int64 // [Rows*Length]; padding is the pad id
	AttentionMask         []int64 // [Rows*Length]; 1 for real tokens
	MarkerPos             []int64 // [Rows*Markers]
	MarkerMask            []bool  // [Rows*Markers]; true for real markers
	QType                 []int64 // [Rows]; Type.Code
	Rows, Length, Markers int
}

// Collate pads rows into one Batch. padID is fed to the encoder under a zero mask, so it must be
// a real vocabulary id.
func Collate(rows []Row, padID int32) (Batch, error) {
	if len(rows) == 0 {
		return Batch{}, ErrEmpty
	}
	length, width := 0, MinMarkerSlots
	for i, r := range rows {
		if len(r.IDs) == 0 {
			return Batch{}, fmt.Errorf("collate: row %d has no token ids", i)
		}
		if !r.Type.Valid() {
			return Batch{}, fmt.Errorf("collate: row %d has question type %q", i, r.Type)
		}
		for _, m := range r.Markers {
			if m < 0 || m >= len(r.IDs) {
				return Batch{}, fmt.Errorf("collate: row %d has a marker at %d outside its %d ids", i, m, len(r.IDs))
			}
		}
		length = max(length, len(r.IDs))
		width = max(width, len(r.Markers))
	}
	b := Batch{
		InputIDs:      make([]int64, len(rows)*length),
		AttentionMask: make([]int64, len(rows)*length),
		MarkerPos:     make([]int64, len(rows)*width),
		MarkerMask:    make([]bool, len(rows)*width),
		QType:         make([]int64, len(rows)),
		Rows:          len(rows), Length: length, Markers: width,
	}
	for i, r := range rows {
		base := i * length
		for j := 0; j < length; j++ {
			if j < len(r.IDs) {
				b.InputIDs[base+j] = int64(r.IDs[j])
				b.AttentionMask[base+j] = 1
			} else {
				b.InputIDs[base+j] = int64(padID)
			}
		}
		for j, m := range r.Markers {
			b.MarkerPos[i*width+j] = int64(m)
			b.MarkerMask[i*width+j] = true
		}
		b.QType[i] = int64(r.Type.Code())
	}
	return b, nil
}
