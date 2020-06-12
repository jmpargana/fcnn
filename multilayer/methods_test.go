package fcnn

import (
	"testing"

	"github.com/jmpargana/matrix"
)

func TestForwProp(t *testing.T) {
	for i := 1; i < 10; i++ {
		in := matrix.New(3, 1)
		expected := i

		nn, err := New([]int{3, 5, 6, 5, 2}, expected, "relu", "relu", 1, 1, 0.5)
		if err != nil {
			t.Errorf("no reason to fail here: %v", err)
		}

		out, err := nn.ForwProp(in)
		if err != nil {
			t.Errorf("no reason to fail here: %v", err)
		}

		if out.NumRows != expected {
			t.Errorf("expected output size to be %d, instead got %d", expected, out.NumRows)
		}
	}
}
