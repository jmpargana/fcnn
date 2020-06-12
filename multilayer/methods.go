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

// BackProp receives an expected output Y, calculates the first delta
// using the quadratic function (might generalize later).
// It then propagates the error further saving the weight and bias errors
// indexing in the member attributes for later use with the Gradient Descent.
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

// calculateDelta receives an error vector and calculates the error of each
// layer. Uses the quadratic function for the outputLayer and normal weight
// transposing for all others.
func (m *MultiLayerPerceptron) calculateDelta(index int, output matrix.Matrix) error {

	var delta matrix.Matrix
	var err error

	if index == len(m.deltas)-1 {
		delta, err = m.outputLayer.BackPropOutLayer(output)
	} else {
		// TODO: need to check one extra for output layer
		delta, err = m.hiddenLayers[index].BackProp(
			m.deltas[index+1],
			m.hiddenLayers[index+2].Weights)
	}

	if err != nil {
		return err
	}

	m.deltas[index] = delta

	return nil
}

// calculateWeight multiplies the previous activatedOutput with the current error
// generating a matrix of weights' errors.
func (m *MultiLayerPerceptron) calculateWeight(index int) error {

	// TODO: check for indices in hidden and output layers
	transposedPrevAct, err := matrix.Trans(m.hiddenLayers[index-1].Output)
	// transposedPrevAct, err := matrix.Trans(m.activations[index-1])
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

// GradientDescent is called after calculating the errors for both the bias and
// weights in each layer. It then sequentially updates both. It does so calling
// the method in each layer which simply subtracts the multiplyed error and learning
// rate from the current value.
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
