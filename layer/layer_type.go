package layer

import "github.com/jmpargana/matrix"

// Layer is an abstraction of an array of perceptrons.
// It contains the weights of all perceptron as well
// as its activation function.
// The output and sum matrices will store the vector calculated
// in the ForwProp needed for the BackProp.
type Layer struct {
	actFn                      string
	weights, output, sum, bias matrix.Matrix
}
