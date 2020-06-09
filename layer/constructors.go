package layer

import (
	"errors"
	"fmt"
	"github.com/jmpargana/fcnn"
	"github.com/jmpargana/matrix"
)

// New creates a layer with a given input and output size.
// It generates a random matrix of weights and allocates memory for
// the sum and output vectors.
// It checks weather the input size is valid and if the activation
// function exists in map.
func New(actFn string, inSize, outSize int) (Layer, error) {
	if _, ok := fcnn.ActivationFunctions[actFn]; !ok {
		return Layer{}, errors.New(fmt.Sprintf("%s is not available as activation function", actFn))
	}

	if inSize < 1 || outSize < 1 {
		return Layer{}, errors.New(fmt.Sprintf("that would be an invalid size for a matrix: %dx%d", inSize, outSize))
	}

	weights := matrix.NewRandom(inSize, outSize)
	output, sum := matrix.New(outSize, 1), matrix.New(outSize, 1)

	return Layer{
		actFn:   actFn,
		weights: weights,
		output:  output,
		sum:     sum,
	}, nil
}
