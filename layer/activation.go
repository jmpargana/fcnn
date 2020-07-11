package layer

import (
	// import to dot to avoid package verbosity

	"math"
	"math/rand"

	"github.com/jmpargana/matrix"
)

type fnType func(matrix.Matrix) (matrix.Matrix, error)

// ActivationFunctions is a collection of the available functions that can be
// used in the back and forward propagation steps.
// If one is not available, the Layer won't be instantiated.
// Their mathematical formulas can be seen in https://en.wikipedia.org/wiki/Activation_function.
var ActivationFunctions = map[string]fnType{
	"relu":        relu,
	"identity":    identity,
	"binary_step": binaryStep,
	"sigmoid":     sigmoid,
	"tanh":        tanH,
	"lrelu":       lReLU,
	"rrelu":       rReLU,
	"arctan":      arcTan,
	"softmax":     softmax,
}

func identity(in matrix.Matrix) (m matrix.Matrix, err error) {
	m, err = apply(in, func(x float64) float64 { return x })
	return
}

func relu(in matrix.Matrix) (m matrix.Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		if x <= 0 {
			return 0
		}
		return x
	})
	return
}

func binaryStep(in matrix.Matrix) (m matrix.Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		if x <= 0 {
			return 0
		}
		return 1
	})
	return
}

func sigmoid(in matrix.Matrix) (m matrix.Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	})
	return
}

func tanH(in matrix.Matrix) (m matrix.Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		return (math.Exp(x) - math.Exp(-x)) / (math.Exp(x) + math.Exp(-x))
	})
	return
}

func lReLU(in matrix.Matrix) (m matrix.Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		if x < 0 {
			return 0.01 * x
		}
		return x
	})
	return
}

func rReLU(in matrix.Matrix) (m matrix.Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		alpha := rand.Float64()
		if x < 0 {
			return alpha * x
		}
		return x
	})
	return
}

func arcTan(in matrix.Matrix) (m matrix.Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		return math.Atan(x)
	})
	return
}

func softmax(in matrix.Matrix) (matrix.Matrix, error) {

	sum := 0.0
	max := math.Inf(-1)
	for row := 0; row < in.NumRows; row++ {
		elem, err := in.Get(row, 0)
		if err != nil {
			return matrix.Matrix{}, err
		}
		if elem > max {
			max = elem
		}
	}

	for row := 0; row < in.NumRows; row++ {
		elem, err := in.Get(row, 0)
		if err != nil {
			return matrix.Matrix{}, err
		}
		sum += math.Exp(elem - max)
	}

	m := matrix.New(in.NumRows, 1)
	for row := 0; row < in.NumRows; row++ {
		elem, _ := in.Get(row, 0)
		if err := m.Set(row, 0, (math.Exp(elem) / sum)); err != nil {
			return matrix.Matrix{}, err
		}
	}
	return m, nil
}
