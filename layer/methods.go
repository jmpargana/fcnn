package layer

import (
	"errors"

	"github.com/jmpargana/matrix"
)

// ForwProp gets called from the MultiLayerPerceptron for each layer
// sequentially and calculates the output saving each stage in the sum and
// output variables, which are needed for the BackProp method.
func (l *Layer) ForwProp(input matrix.Matrix) (matrix.Matrix, error) {
	if input.NumRows != l.InSize() || input.NumCols != 1 {
		return matrix.Matrix{}, errors.New("can't perform forward propagation with invalid input size")
	}

	l.Sum, _ = matrix.Mult(l.Weights, input)
	if err := l.Sum.Add(l.Bias); err != nil {
		return matrix.Matrix{}, err
	}
	l.Output, _ = ActivationFunctions[l.actFn](l.Sum)

	return l.Output, nil
}

// BackProp implements the backpropagation algorithm for any layer.
// It takes the delta loss and weights from layer (l+1) and calculates the error of
// the current layer.
func (l *Layer) BackProp(prevDelta, weights matrix.Matrix) (matrix.Matrix, error) {
	// TODO: in order to prevent allocation everytime this functions is called,
	// maybe it would be clever to allocate in the constructor
	transposedWeights, _ := matrix.Trans(weights)

	weightedErrors, err := matrix.Mult(transposedWeights, prevDelta)
	if err != nil {
		return matrix.Matrix{}, err
	}

	derivative, _ := DerivativeFunctions[l.actFn](l.Sum)

	delta, err := matrix.HadamardProd(weightedErrors, derivative)
	if err != nil {
		return matrix.Matrix{}, err
	}

	return delta, nil
}

// BackPropOutLayer is the special case for the backpropagation algorithm, when performed
// starting with the output layer.
// It is very similar to the normal one if the cost function used is the quadratic function.
// TODO: It will stay as a seperate function to make the refactoring process easier
// when generalising to any cost function.
func (l *Layer) BackPropOutLayer(expected matrix.Matrix) (matrix.Matrix, error) {

	computedError, err := matrix.Sub(l.Output, expected)
	if err != nil {
		return matrix.Matrix{}, err
	}

	// these two can't fail, since the dimensions are already checked before
	derivative, _ := DerivativeFunctions[l.actFn](l.Sum)
	delta, _ := matrix.HadamardProd(computedError, derivative)

	return delta, nil
}

// UpdateWeights receives the multiplyed learning rate and error and is subtracted
// from the layers weight matrix.
func (l *Layer) UpdateWeights(derived matrix.Matrix) error {
	return l.Weights.Sub(derived)
}

// UpdateBias receives the multiplyed learning rate and error and is subtracted
// from the layers bias vector.
func (l *Layer) UpdateBias(derived matrix.Matrix) error {
	return l.Bias.Sub(derived)
}
