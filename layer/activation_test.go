package layer

import (
	"testing"

	"github.com/jmpargana/matrix"
)

func TestAvailableFn(t *testing.T) {
	fns := []string{"relu", "identity", "binary_step", "sigmoid", "tanh", "lrelu", "rrelu", "arctan", "softmax"}

	for _, fn := range fns {
		if _, ok := ActivationFunctions[fn]; !ok {
			t.Errorf("%s is not available in list", fn)
		}
	}
}

func TestNonAvailableFn(t *testing.T) {
	invalidFns := []string{"logistic", "softsign", "isru", "plu", "gelu", "brelu", "srelu", "apl"}

	for _, fn := range invalidFns {
		if _, ok := ActivationFunctions[fn]; ok {
			t.Errorf("%s is not supposed to be available", fn)
		}
	}
}

func TestIdentity(t *testing.T) {
	for _, inOut := range identityTest {
		in, out := matrix.NewFrom(inOut.in), matrix.NewFrom(inOut.out)
		c, err := identity(in)
		if err != nil {
			t.Errorf("shouldn't have failed: %v", err)
		}

		if !out.Equal(c) {
			t.Errorf("%v should be equal to %v", out, c)
		}
	}
}

func TestBinaryStep(t *testing.T) {
	for _, inOut := range binaryStepTest {
		in, out := matrix.NewFrom(inOut.in), matrix.NewFrom(inOut.out)
		c, err := binaryStep(in)
		if err != nil {
			t.Errorf("shouldn't have failed: %v", err)
		}

		if !out.Equal(c) {
			t.Errorf("%v should be equal to %v", out, c)
		}
	}
}

func TestReLU(t *testing.T) {
	for _, inOut := range reluTest {
		in, out := matrix.NewFrom(inOut.in), matrix.NewFrom(inOut.out)
		c, err := relu(in)
		if err != nil {
			t.Errorf("shouldn't have failed: %v", err)
		}

		if !out.Equal(c) {
			t.Errorf("%v should be equal to %v", out, c)
		}
	}
}

func TestLReLU(t *testing.T) {
	for _, inOut := range lReLUTest {
		in, out := matrix.NewFrom(inOut.in), matrix.NewFrom(inOut.out)
		c, err := lReLU(in)
		if err != nil {
			t.Errorf("shouldn't have failed: %v", err)
		}

		if !out.Equal(c) {
			t.Errorf("%v should be equal to %v", out, c)
		}
	}
}
