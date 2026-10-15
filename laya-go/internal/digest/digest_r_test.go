package digest

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"
)

func TestFileRefusesTrailingSlash(t *testing.T) {
	fsys := fstest.MapFS{"a/b.bin": {Data: []byte("graph")}}
	err := File(fsys, "a/b.bin/", hexOf("graph"))
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("err = %v", err)
	}
}

func TestFileRefusesNonRegularFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.Symlink("/dev/zero", filepath.Join(dir, "zero.onnx")); err != nil {
		t.Fatal(err)
	}
	err := File(os.DirFS(dir), "zero.onnx", hexOf("x"))
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("err = %v", err)
	}
}

func TestFromEnv(t *testing.T) {
	for _, c := range []struct {
		name, val string
		want      map[string]string
		bad       bool
	}{
		{"unset", "", nil, false},
		{"blank", "  \n", nil, false},
		{"object", ` {"a":"b"} `, map[string]string{"a": "b"}, false},
		{"not json", "nope", nil, true},
		{"array", `["a"]`, nil, true},
		{"non-string", `{"a":1}`, nil, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv(Env, c.val)
			got, err := FromEnv()
			if c.bad != (err != nil) {
				t.Fatalf("err = %v", err)
			}
			if len(got) != len(c.want) || got["a"] != c.want["a"] {
				t.Fatalf("got %v", got)
			}
		})
	}
}
