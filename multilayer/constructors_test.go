package fcnn

import (
	"testing"
)

func TestHiddenLayers(t *testing.T) {
	for _, hl := range hiddenLayersTest {
		hlayers, err := startHiddenLayers("relu", hl)
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		if len(hlayers) != len(hl)-1 {
			t.Errorf("the hidden layers list contains values of in and output sizes, so it should create one layer less")
		}

		for i := 0; i < len(hl)-1; i++ {
			if hlayers[i].InSize() != hl[i] {
				t.Errorf("in size of current layer should be: %d, instead got: %d", hl[i], hlayers[i].InSize())
			}

			if hlayers[i].OutSize() != hl[i+1] {
				t.Errorf("out size of current layer should be: %d, instead got: %d", hl[i+1], hlayers[i].OutSize())
			}
		}
	}
}
