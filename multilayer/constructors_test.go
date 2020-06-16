package multilayer

import (
	"testing"
)

func TestHiddenLayers(t *testing.T) {
	for _, hl := range HiddenLayersTest {
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

func TestInvalidNewForZeroHiddenLayers(t *testing.T) {
	_, err := New([]int{1}, 1, "relu", "relu", 1, 1, 0.5)
	if err == nil {
		t.Errorf("need at least 1 hidden layer, otherwise act function for hidden layer doesn't make sense")
	}
}

func TestInvalidHiddenLayers(t *testing.T) {
	for _, hl := range invalidHiddenLayersTest {
		_, err := startHiddenLayers("relu", hl)
		if err == nil {
			t.Errorf("shouldn have failed here")
		}
	}
}

func TestInvalidLearningRate(t *testing.T) {
	_, err := New([]int{1, 1}, 1, "relu", "relu", 1, 1, 0)
	if err == nil {
		t.Errorf("learning rate should be between 0 and 1")
	}

	_, err = New([]int{1, 1}, 1, "relu", "relu", 1, 1, 1)
	if err == nil {
		t.Errorf("learning rate should be between 0 and 1")
	}
}

func TestInvalidBatchEpochSizes(t *testing.T) {
	_, err := New([]int{1, 1}, 1, "relu", "relu", 0, 1, 0.5)
	if err == nil {
		t.Errorf("shouldn't be able to build with batch size 0")
	}

	_, err = New([]int{1, 1}, 1, "relu", "relu", 1, 0, 0.5)
	if err == nil {
		t.Errorf("shouldn't be able to build with epoch size 0")
	}

	_, err = New([]int{1, 1}, 1, "relu", "relu", -3, 1, 0.5)
	if err == nil {
		t.Errorf("shouldn't be able to build with negative batch size")
	}
	_, err = New([]int{1, 1}, 1, "relu", "relu", 1, -11, 0.5)
	if err == nil {
		t.Errorf("shouldn't be able to build with negative epoch size")
	}

	_, err = New([]int{1, 1}, 1, "relu", "unavailable", 1, 1, 0.5)
	if err == nil {
		t.Errorf("shouldn't be able to build with non valid activation function")
	}
}
