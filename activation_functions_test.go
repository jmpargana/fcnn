package fcnn

import "testing"

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

}
