package fcnn

import (
	"testing"

	"github.com/jmpargana/fcnn/layer"
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

func TestInvalidForwProp(t *testing.T) {
	in := matrix.New(4, 1)
	nn, _ := New([]int{1, 2}, 2, "relu", "relu", 1, 1, 0.5)
	_, err := nn.ForwProp(in)
	if err == nil {
		t.Errorf("should only accept right input size")
	}
}

func TestCalculateDeltaMiddle(t *testing.T) {

}

func TestCalculateDeltaLastHidden(t *testing.T) {

}

func TestCalculateDeltaOut(t *testing.T) {
	for _, test := range deltaOutTest {
		output := matrix.NewFrom(test.output)
		expected := matrix.NewFrom(test.expected)
		sum := matrix.NewFrom(test.sum)

		nn, _ := New([]int{1, 1}, output.NumRows, "relu", test.actFn, 1, 1, 0.5)
		nn.outputLayer.Sum = sum
		nn.outputLayer.Output, _ = layer.ActivationFunctions[test.actFn](sum)

		if err := nn.calculateDelta(1, output); err != nil {
			t.Errorf("not supposed to fail here: %v", err)
		}

		if !matrix.Equal(expected, nn.deltas[1]) {
			t.Errorf("received:\n%v\nhad sum:\n%v\nand output:\n%v\ngot:\n%v\nexpected:\n%v\n", output, sum, nn.outputLayer.Output, nn.deltas[1], expected)
		}
	}
}
