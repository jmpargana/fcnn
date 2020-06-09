package fcnn

import (
	. "github.com/jmpargana/matrix"
	"math"
	"math/rand"
)

// DerivativeFunctions is a map of the derivatives of the available activation functions.
// It uses the formulas defined in https://en.wikipedia.org/wiki/Activation_function.
var DerivativeFunctions = map[string]fnType{
	"relu":        dRelu,
	"identity":    dIdentity,
	"binary_step": dBinaryStep,
	"sigmoid":     dSigmoid,
	"tanh":        dTanH,
	"lrelu":       dLReLU,
	"rrelu":       dRReLU,
	"arctan":      dArcTan,
	"softmax":     dSoftmax,
}

func dIdentity(in Matrix) (m Matrix, err error) {
	m, err = apply(in, func(x float64) float64 { return 1 })
	return
}

func dRelu(in Matrix) (m Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		if x <= 0 {
			return 0
		}
		return x
	})
	return
}

func dBinaryStep(in Matrix) (m Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		if x != 0 {
			return 0
		}
		// ? https://en.wikipedia.org/wiki/Activation_function
		return 1
	})
	return
}

func dSigmoid(in Matrix) (m Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		fx := 1 / (1 + math.Exp(-x))
		return fx * (1 - fx)
	})
	return
}

func dTanH(in Matrix) (m Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		return (math.Exp(x) - math.Exp(-x)) / (math.Exp(x) + math.Exp(-x))
	})
	return
}

func dLReLU(in Matrix) (m Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		fx := x
		if x < 0 {
			fx = 0.01 * x
		}
		return 1 - fx*fx
	})
	return
}

func dRReLU(in Matrix) (m Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		alpha := rand.Float64()
		if x < 0 {
			return alpha
		}
		return 1
	})
	return
}

func dArcTan(in Matrix) (m Matrix, err error) {
	m, err = apply(in, func(x float64) float64 {
		return 1 / (x*x + 1)
	})
	return
}

// TODO: implement
func dSoftmax(in Matrix) (Matrix, error) {
	return Matrix{}, nil
}
