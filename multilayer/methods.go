package fcnn

import (
	"github.com/jmpargana/matrix"
)

// ForwProp calls the ForwProp method for each layer sequentially saving its output
// in the in vector each time.
// It returns an output vector which gets compared with the expected value.
func (m *MultiLayerPerceptron) ForwProp(in matrix.Matrix) (matrix.Matrix, error) {

	for _, layer := range m.hiddenLayers {

		var err error
		in, err = layer.ForwProp(in)
		if err != nil {
			return matrix.Matrix{}, err
		}
	}

	// Maybe output layer should be part of the same slice of layers from
	// the hidden layers
	in, err := m.outputLayer.ForwProp(in)
	if err != nil {
		return matrix.Matrix{}, err
	}

	return in, nil
}
