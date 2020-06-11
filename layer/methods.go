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

	l.sum, _ = matrix.Mult(l.weights, input)
	if err := l.sum.Add(l.bias); err != nil {
		return matrix.Matrix{}, err
	}
	l.output, _ = ActivationFunctions[l.actFn](l.sum)

	return l.output, nil
}

// BackProp implements the backpropagation algorithm for any layer.
// It takes the delta loss from layer (l+1) and calculates the error of
// the current layer.
func (l *Layer) BackProp(loss matrix.Matrix) (matrix.Matrix, error) {
	// in order to prevent allocation everytime this functions is called,
	// maybe it would be clever to allocate in the constructor
	transposedWeights, err := matrix.Trans(l.weights)
	if err != nil {
		return matrix.Matrix{}, err
	}

	weightedErrors, err := matrix.Mult(transposedWeights, loss)
	if err != nil {
		return matrix.Matrix{}, err
	}

	derivative, _ := DerivativeFunctions[l.actFn](l.sum)

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

	computedError, err := matrix.Sub(l.output, expected)
	if err != nil {
		return matrix.Matrix{}, err
	}

	derivative, _ := DerivativeFunctions[l.actFn](l.sum)

	delta, err := matrix.HadamardProd(computedError, derivative)
	if err != nil {
		return matrix.Matrix{}, err
	}

	return delta, nil
}

func (l *Layer) UpdateWeights(derived matrix.Matrix) error {
	return l.weights.Sub(derived)
}

func (l *Layer) UpdateBias(derived matrix.Matrix) error {
	return l.bias.Sub(derived)
}
