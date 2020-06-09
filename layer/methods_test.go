package layer

import (
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

}
