package layer

import (
	_ "fmt"
	"testing"

	"github.com/jmpargana/matrix"
)

func TestForwPropInvalidInput(t *testing.T) {
	layer, err := New("relu", 8, 4)
	if err != nil {
		t.Errorf("should not fail here: %v", err)
	}

	in := matrix.New(5, 1)
	_, err = layer.ForwProp(in)
	if err == nil {
		t.Errorf("should not be able to perform formward propagation with non matching sizes")
	}
}

func TestForwPropInvalidInput2(t *testing.T) {
	layer, err := New("relu", 8, 4)
	if err != nil {
		t.Errorf("should not fail here: %v", err)
	}

	in := matrix.New(5, 2)
	_, err = layer.ForwProp(in)
	if err == nil {
		t.Errorf("should not be able to perform formward propagation with non vector input sizes")
	}
}

func TestForwProp(t *testing.T) {
	for _, fp := range forwPropsTest {
		layer, err := New(fp.actFn, len(fp.in), len(fp.out))
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		in, sum, out := matrix.NewFrom(fp.in), matrix.NewFrom(fp.sum), matrix.NewFrom(fp.out)

		weights := matrix.NewFrom(fp.weights)
		layer.Weights = weights

		got, err := layer.ForwProp(in)
		if err != nil {
			t.Errorf("failed in forward propagation: %v", err)
		}

		if !layer.sum.Equal(sum) {
			t.Errorf("\n%vsum matrix should be equal to\n%v", layer.sum, sum)
		}

		if !got.Equal(out) {
			t.Errorf("\n%vshould be equal to\n%v", got, out)
		}
	}
}

func TestForwPropWithBias(t *testing.T) {
	for _, fp := range forwPropsWithBiasTest {
		layer, err := New(fp.actFn, len(fp.in), len(fp.out))
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		in, sum, out, bias := matrix.NewFrom(fp.in), matrix.NewFrom(fp.sum), matrix.NewFrom(fp.out), matrix.NewFrom(fp.bias)

		weights := matrix.NewFrom(fp.weights)
		layer.Weights = weights
		layer.bias = bias

		got, err := layer.ForwProp(in)
		if err != nil {
			t.Errorf("failed in forward propagation: %v", err)
		}

		if !layer.sum.Equal(sum) {
			t.Errorf("\n%vsum matrix should be equal to\n%v", layer.sum, sum)
		}

		if !got.Equal(out) {
			t.Errorf("\n%vshould be equal to\n%v", got, out)
		}
	}
}

func TestForwPropInvalidBias1(t *testing.T) {
	layer, err := New("relu", 20, 29)
	if err != nil {
		t.Errorf("shouldn't fail here: %v", err)
	}

	in := matrix.New(20, 1)
	layer.Weights = matrix.New(1, 20)

	if _, err := layer.ForwProp(in); err == nil {
		t.Errorf("non matching input size and weights")
	}
}

func TestForwPropInvalidBias2(t *testing.T) {
	layer, err := New("relu", 20, 29)
	if err != nil {
		t.Errorf("shouldn't fail here: %v", err)
	}

	in := matrix.New(20, 1)
	layer.bias = matrix.New(15, 15)

	if _, err = layer.ForwProp(in); err == nil {
		t.Errorf("non matching input size and weights")
	}
}

func TestUpdateWeights(t *testing.T) {
	for _, mats := range updateWeightsTest {
		weights := matrix.NewFrom(mats.in)
		derivedError := matrix.NewFrom(mats.out)
		expected := matrix.NewFrom(mats.exp)

		layer, err := New("relu", 1, 1)
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		layer.Weights = weights

		if err := layer.UpdateWeights(derivedError); err != nil {
			t.Errorf("shoudln't fail here: %v", err)
		}

		if !layer.Weights.Equal(expected) {
			t.Errorf("\nweights:\n%vminues error:\n%vshould equal:\n%vinstead got:\n%v", weights, derivedError, expected, layer.Weights)
		}
	}
}

func TestUpdateWeightsInvalid(t *testing.T) {
	for _, mats := range updateWeightsTestInvalid {
		weights := matrix.NewFrom(mats.in)
		derivedError := matrix.NewFrom(mats.out)

		layer, err := New("relu", 1, 1)
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		layer.Weights = weights

		if err := layer.UpdateWeights(derivedError); err == nil {
			t.Errorf("not supposed to be able to subtract %v from %v", derivedError, weights)
		}
	}
}

func TestUpdateBias(t *testing.T) {
	for _, mats := range updateBiasTest {
		bias := matrix.NewFrom(mats.in)
		derivedError := matrix.NewFrom(mats.out)
		expected := matrix.NewFrom(mats.exp)

		layer, err := New("relu", 1, 1)
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		layer.bias = bias

		if err := layer.UpdateBias(derivedError); err != nil {
			t.Errorf("shoudln't fail here: %v", err)
		}

		if !layer.bias.Equal(expected) {
			t.Errorf("\nbias:\n%vminus derived error:\n%vshould equal:\n%vinstead got:\n%v", bias, derivedError, expected, layer.bias)
		}
	}
}

func TestUpdateBiasInvalid(t *testing.T) {
	for _, mats := range updateBiasTestInvalid {
		bias := matrix.NewFrom(mats.in)
		derivedError := matrix.NewFrom(mats.out)

		layer, err := New("relu", 1, 1)
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		layer.bias = bias

		if err := layer.UpdateBias(derivedError); err == nil {
			t.Errorf("not supposed to be able to subtract %v from %v", derivedError, bias)
		}
	}
}

func TestBackPropOutLayer(t *testing.T) {
	for _, test := range backPropOutLayerTest {
		desiredOutput := matrix.NewFrom(test.desiredOutput)
		expectedDelta := matrix.NewFrom(test.expectedDelta)
		sum := matrix.NewFrom(test.sum)

		layer, _ := New(test.actFn, 1, 1)
		layer.sum = sum
		layer.Output, _ = ActivationFunctions[test.actFn](layer.sum)

		delta, err := layer.BackPropOutLayer(desiredOutput)
		if err != nil {
			t.Errorf("shouln't fail here: %v", err)
		}

		if !expectedDelta.Equal(delta) {
			t.Errorf("\nsum:\n%v\nactFn:%s\ninput:\n%vexpected:\n%vgot:\n%v", sum, test.actFn, desiredOutput, expectedDelta, delta)
		}
	}
}

func TestBackPropOutLayerInvalid(t *testing.T) {
	for _, test := range backPropOutLayerInvalidTest {
		desiredOutput := matrix.NewFrom(test.desiredOutput)
		sum := matrix.NewFrom(test.sum)

		layer, _ := New(test.actFn, 1, 1)
		layer.sum = sum

		if _, err := layer.BackPropOutLayer(desiredOutput); err == nil {
			t.Errorf("should fail here")
		}
	}
}

func TestBackProp(t *testing.T) {
	for _, test := range backPropTest {
		deltaPlus1 := matrix.NewFrom(test.deltaPlus1)
		weights := matrix.NewFrom(test.weights)
		delta := matrix.NewFrom(test.expected)
		sum := matrix.NewFrom(test.sum)

		layer, err := New(test.actFn, 1, 1)
		if err != nil {
			t.Errorf("shouldn't be failing here")
		}

		layer.sum = sum

		got, err := layer.BackProp(deltaPlus1, weights)
		if err != nil {
			t.Errorf("not supposed to fail here: %v", err)
		}

		if !got.Equal(delta) {
			t.Errorf("weights:\n%v\ndelta+1:\n%v\nsum:\n%v\ngot:\n%v\nexpected:\n%v\n", weights, deltaPlus1, sum, got, delta)
		}
	}
}

func TestBackPropInvalid(t *testing.T) {
	for _, test := range backPropTestInvalid {
		deltaPlus1 := matrix.NewFrom(test.deltaPlus1)
		weights := matrix.NewFrom(test.weights)
		sum := matrix.NewFrom(test.sum)

		layer, err := New(test.actFn, 1, 1)
		if err != nil {
			t.Errorf("shouldn't be failing here")
		}

		layer.sum = sum

		if _, err := layer.BackProp(deltaPlus1, weights); err == nil {
			t.Errorf("was supposed to fail here")
		}
	}
}
