// Package digest checks SHA-256 digests the way laya's verify_digests does: streamed, so a
// multi-gigabyte graph is never read whole, and compared case-insensitively.
package digest

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"strings"
)

// ErrMismatch is wrapped by the error for a file whose digest is not the expected one.
var ErrMismatch = errors.New("SHA-256 mismatch")

// ErrUnsafePath is wrapped by the error for a name that is absolute or escapes its directory.
var ErrUnsafePath = errors.New("unsafe path in expected digests")

// Sum streams r through SHA-256 and returns the lower-case hex digest.
func Sum(r io.Reader) (string, error) {
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Compare is nil when got equals want, ignoring case and surrounding space.
func Compare(name, got, want string) error {
	if strings.EqualFold(got, strings.TrimSpace(want)) {
		return nil
	}
	return fmt.Errorf("laya: %w for %s: expected %s, got %s. The artifact does not match the reviewed "+
		"digest; refusing to load it", ErrMismatch, name, want, got)
}

// Safe returns rel as a clean slash path under its directory, or an error wrapping ErrUnsafePath
// when it is empty, absolute (POSIX or a Windows drive or UNC root) or climbs out with "..".
func Safe(rel string) (string, error) {
	n := strings.ReplaceAll(rel, "\\", "/")
	drive := len(n) >= 2 && n[1] == ':' && (n[0]|0x20 >= 'a' && n[0]|0x20 <= 'z')
	if strings.HasPrefix(n, "/") || drive {
		return "", fmt.Errorf("laya: %w: absolute path %q", ErrUnsafePath, rel)
	}
	if n == "" || n == ".." || strings.HasPrefix(n, "../") || strings.Contains(n, "/../") || strings.HasSuffix(n, "/..") {
		return "", fmt.Errorf("laya: %w: %q", ErrUnsafePath, rel)
	}
	if n = path.Clean(n); n == "." {
		return "", fmt.Errorf("laya: %w: %q", ErrUnsafePath, rel)
	}
	return n, nil
}

// File hashes the regular file rel in fsys and compares it with want. A file that is absent or
// is a directory is an error wrapping fs.ErrNotExist.
func File(fsys fs.FS, rel, want string) error {
	n, err := Safe(rel)
	if err != nil {
		return err
	}
	f, err := fsys.Open(n)
	if err != nil {
		return fmt.Errorf("laya: cannot verify %q: %w", rel, err)
	}
	defer f.Close()
	if st, err := f.Stat(); err != nil || st.IsDir() {
		return fmt.Errorf("laya: cannot verify %q: not a file: %w", rel, fs.ErrNotExist)
	}
	got, err := Sum(f)
	if err != nil {
		return fmt.Errorf("laya: cannot verify %q: %w", rel, err)
	}
	return Compare(rel, got, want)
}
