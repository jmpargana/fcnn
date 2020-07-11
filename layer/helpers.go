package layer

import (
	"github.com/jmpargana/matrix"
)

// apply is called in most of the activation functions and its derivatives.
func apply(in matrix.Matrix, f func(float64) float64) (matrix.Matrix, error) {
	res := matrix.New(in.NumRows, 1)

	for row := 0; row < in.NumRows; row++ {
		elem, err := in.Get(row, 0)
		if err != nil {
			return matrix.Matrix{}, nil
		}
		res.Set(row, 0, f(elem))
	}
	return res, nil
}
