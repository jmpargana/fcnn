package fcnn

import (
	"github.com/jmpargana/matrix"
)

// ForwProp calls the ForwProp method for each layer sequentially saving its output
// in the in vector each time.
// It returns an output vector which gets compared with the expected value.
func (m *MultiLayerPerceptron) ForwProp(in matrix.Matrix) (matrix.Matrix, error) {

	for i, layer := range m.hiddenLayers {

		var err error
		in, err = layer.ForwProp(in)
		if err != nil {
			return matrix.Matrix{}, err
		}

		m.activations[i] = in
	}

	// Maybe output layer should be part of the same slice of layers from
	// the hidden layers
	in, err := m.outputLayer.ForwProp(in)
	if err != nil {
		return matrix.Matrix{}, err
	}

	m.activations[len(m.hiddenLayers)] = in

	return in, nil
}

func (m *MultiLayerPerceptron) BackProp(output matrix.Matrix) error {

	for i := len(m.deltas) - 1; i > 1; i-- {

		if err := m.calculateDelta(i, output); err != nil {
			return err
		}

		if err := m.calculateWeight(i); err != nil {
			return err
		}
	}

	// TODO: deal with first layer

	return nil
}

func (m *MultiLayerPerceptron) calculateDelta(index int, output matrix.Matrix) error {

	var delta matrix.Matrix
	var err error

	if index == len(m.deltas)-1 {
		delta, err = m.outputLayer.BackPropOutLayer(output)
	} else {
		delta, err = m.hiddenLayers[index].BackProp(m.deltas[index+1])
	}

	if err != nil {
		return err
	}

	m.deltas[index] = delta

	return nil
}

func (m *MultiLayerPerceptron) calculateWeight(index int) error {

	transposedPrevAct, err := matrix.Trans(m.activations[index-1])
	if err != nil {
		return err
	}
	weight, err := matrix.Mult(m.deltas[index], transposedPrevAct)
	if err != nil {
		return err
	}

	m.weights[index] = weight

	return nil
}

func (m *MultiLayerPerceptron) GradientDescent() error {

	for i, layer := range m.hiddenLayers {

		m.weights[i].MultScalar(m.learningRate)
		if err := layer.UpdateWeights(m.weights[i]); err != nil {
			return err
		}

		m.deltas[i].MultScalar(m.learningRate)
		if err := layer.UpdateBias(m.deltas[i]); err != nil {
			return err
		}
	}

	// TODO: the same with last layer

	return nil
}
