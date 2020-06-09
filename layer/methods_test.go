package layer

import (
	_ "fmt"
	"github.com/jmpargana/matrix"
	"testing"
)

func TestForwPropInvalidInput(t *testing.T) {
	layer, err := New("relu", 8, 4)
	if err != nil {
		t.Errorf("should not fail here: %v", err)
	}

	in := matrix.New(5, 1)
	_, err = layer.ForwProp(in)
	if err == nil {
		t.Errorf("should not be able to perform formward propagation with non matching sizes")
	}
}

func TestForwPropInvalidInput2(t *testing.T) {
	layer, err := New("relu", 8, 4)
	if err != nil {
		t.Errorf("should not fail here: %v", err)
	}

	in := matrix.New(5, 2)
	_, err = layer.ForwProp(in)
	if err == nil {
		t.Errorf("should not be able to perform formward propagation with non vector input sizes")
	}
}

func TestForwProp(t *testing.T) {
	for _, fp := range forwPropsTest {
		layer, err := New(fp.actFn, len(fp.in), len(fp.out))
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		in, sum, out := matrix.NewFrom(fp.in), matrix.NewFrom(fp.sum), matrix.NewFrom(fp.out)

		weights := matrix.NewFrom(fp.weights)
		layer.weights = weights

		got, err := layer.ForwProp(in)
		if err != nil {
			t.Errorf("failed in forward propagation: %v", err)
		}

		if !layer.sum.Equal(sum) {
			t.Errorf("\n%vsum matrix should be equal to\n%v", layer.sum, sum)
		}

		if !got.Equal(out) {
			t.Errorf("\n%vshould be equal to\n%v", got, out)
		}
	}
}
