package layer

import (
	// "github.com/jmpargana/matrix"
	"testing"
)

func TestNewLayer(t *testing.T) {
	for _, l := range layersTest {
		layer, err := New(l.actFn, l.inSize, l.outSize)
		if err != nil {
			t.Errorf("something went wrong: %v", err)
		}

		if layer.sum.NumCols != 1 || layer.output.NumCols != 1 {
			t.Errorf("the sum and output matrices should be vectors")
		}

		if layer.sum.NumRows != layer.output.NumRows {
			t.Errorf("sum and output vector should have the same size")
		}

		if layer.sum.NumRows != layer.weights.NumCols {
			t.Errorf("sum and weight matrix should have the matching sizes")
		}
	}
}

func TestNewLayerInvalid(t *testing.T) {
	for _, l := range layersInvalidTest {
		_, err := New(l.actFn, l.inSize, l.outSize)
		if err == nil {
			t.Errorf("shouldn't be able to build layer with this: %v", l)
		}
	}
}
