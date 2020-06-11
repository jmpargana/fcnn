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

// TODO: implement
func (l *Layer) BackProp(loss matrix.Matrix) (matrix.Matrix, error) {
	return matrix.Matrix{}, nil
}
