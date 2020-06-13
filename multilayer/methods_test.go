package fcnn

import (
	"testing"

	"github.com/jmpargana/fcnn/layer"
	"github.com/jmpargana/matrix"
)

func TestForwProp(t *testing.T) {
	for i := 1; i < 10; i++ {
		in := matrix.New(3, 1)
		expected := i

		nn, err := New([]int{3, 5, 6, 5, 2}, expected, "relu", "relu", 1, 1, 0.5)
		if err != nil {
			t.Errorf("no reason to fail here: %v", err)
		}

		out, err := nn.ForwProp(in)
		if err != nil {
			t.Errorf("no reason to fail here: %v", err)
		}

		if out.NumRows != expected {
			t.Errorf("expected output size to be %d, instead got %d", expected, out.NumRows)
		}
	}
}

func TestInvalidForwProp(t *testing.T) {
	in := matrix.New(4, 1)
	nn, _ := New([]int{1, 2}, 2, "relu", "relu", 1, 1, 0.5)
	_, err := nn.ForwProp(in)
	if err == nil {
		t.Errorf("should only accept right input size")
	}
}

func TestCalculateDeltaMiddle(t *testing.T) {
	for _, test := range deltaTest {
		delta := matrix.NewFrom(test.delta)
		matrixPlus1 := matrix.NewFrom(test.matrixPlus1)
		prevDelta := matrix.NewFrom(test.prevDelta)
		sum := matrix.NewFrom(test.sum)

		nn, _ := New(test.hiddenLayers, 1, test.actFn, "relu", 1, 1, 0.5)
		nn.hiddenLayers[len(nn.hiddenLayers)-2].Sum = sum
		nn.hiddenLayers[len(nn.hiddenLayers)-1].Weights = matrixPlus1
		nn.deltas[len(nn.hiddenLayers)-1] = prevDelta

		if err := nn.calculateDelta(len(nn.hiddenLayers)-2, matrix.Matrix{}); err != nil {
			t.Errorf("not supposed to fail here: %v", err)
		}

		if !nn.deltas[len(nn.deltas)-3].Equal(delta) {
			t.Errorf("\nprevDelta:\n%v\nsum:\n%v\nprevMatrix:\n%v\ngot:\n%v\nexpected:\n%v\n", prevDelta, sum, matrixPlus1, nn.deltas[len(nn.deltas)-2], delta)
		}
	}
}

func TestCalculateDeltaLastHidden(t *testing.T) {
	for _, test := range deltaLastHiddenTest {
		delta := matrix.NewFrom(test.delta)
		matrixPlus1 := matrix.NewFrom(test.matrixPlus1)
		prevDelta := matrix.NewFrom(test.prevDelta)
		sum := matrix.NewFrom(test.sum)

		nn, _ := New(test.hiddenLayers, matrixPlus1.NumRows, test.actFn, "relu", 1, 1, 0.5)
		nn.hiddenLayers[len(nn.hiddenLayers)-1].Sum = sum
		nn.outputLayer.Weights = matrixPlus1
		nn.deltas[len(nn.hiddenLayers)] = prevDelta

		if err := nn.calculateDelta(len(nn.hiddenLayers)-1, matrix.Matrix{}); err != nil {
			t.Errorf("not supposed to fail here: %v", err)
		}

		if !nn.deltas[len(nn.deltas)-2].Equal(delta) {
			t.Errorf("\nprevDelta:\n%v\nsum:\n%v\nprevMatrix:\n%v\ngot:\n%v\nexpected:\n%v\n", prevDelta, sum, matrixPlus1, nn.deltas[len(nn.deltas)-2], delta)
		}
	}
}

func TestCalculateDeltaOut(t *testing.T) {
	for _, test := range deltaOutTest {
		output := matrix.NewFrom(test.output)
		expected := matrix.NewFrom(test.expected)
		sum := matrix.NewFrom(test.sum)

		nn, _ := New([]int{1, 1}, output.NumRows, "relu", test.actFn, 1, 1, 0.5)
		nn.outputLayer.Sum = sum
		nn.outputLayer.Output, _ = layer.ActivationFunctions[test.actFn](sum)

		if err := nn.calculateDelta(1, output); err != nil {
			t.Errorf("not supposed to fail here: %v", err)
		}

		if !nn.deltas[1].Equal(expected) {
			t.Errorf("received:\n%v\nhad sum:\n%v\nand output:\n%v\ngot:\n%v\nexpected:\n%v\n", output, sum, nn.outputLayer.Output, nn.deltas[1], expected)
		}
	}
}

func TestFailCalculateDelta(t *testing.T) {
	for _, test := range deltaFailTest {
		matrixPlus1 := matrix.NewFrom(test.matrixPlus1)
		prevDelta := matrix.NewFrom(test.prevDelta)
		sum := matrix.NewFrom(test.sum)

		nn, _ := New(test.hiddenLayers, 1, test.actFn, "relu", 1, 1, 0.5)
		nn.hiddenLayers[len(nn.hiddenLayers)-2].Sum = sum
		nn.hiddenLayers[len(nn.hiddenLayers)-1].Weights = matrixPlus1
		nn.deltas[len(nn.hiddenLayers)-1] = prevDelta

		if err := nn.calculateDelta(len(nn.hiddenLayers)-2, matrix.Matrix{}); err == nil {
			t.Errorf("supposed to fail here!")
		}
	}
}

func TestUpdateBias(t *testing.T) {
	for _, test := range updateBiasTest {
		goErr := make(chan error)
		index := test.index
		delta := matrix.NewFrom(test.delta)
		bias := matrix.NewFrom(test.biasWeights)
		expected := matrix.NewFrom(test.expected)

		nn, _ := New(test.hiddenLayers, 1, test.actFn, "relu", 1, 1, test.learningRate)
		nn.deltas[index] = delta
		nn.hiddenLayers[index].Bias = bias

		go nn.updateBias(index, goErr)

		err, _ := <-goErr
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		if !expected.Equal(nn.hiddenLayers[index].Bias) {
			t.Errorf("\nbias:\n%v\ndelta:\n%v\nmultiplyed by learning rate:%f\nshould be:\n%v\ngot:\n%v", delta, bias, test.learningRate, expected, nn.hiddenLayers[index].Bias)
		}
	}
}

func TestUpdateBiasOut(t *testing.T) {
	for _, test := range updateBiasOutTest {
		goErr := make(chan error)
		delta := matrix.NewFrom(test.delta)
		bias := matrix.NewFrom(test.biasWeights)
		expected := matrix.NewFrom(test.expected)

		nn, _ := New(test.hiddenLayers, test.hiddenLayers[len(test.hiddenLayers)-1], test.actFn, "relu", 1, 1, test.learningRate)
		nn.deltas[len(nn.deltas)-1] = delta
		nn.outputLayer.Bias = bias

		go nn.updateBias(len(nn.deltas)-1, goErr)

		err, _ := <-goErr
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		if !expected.Equal(nn.outputLayer.Bias) {
			t.Errorf("\nbias:\n%v\ndelta:\n%v\nmultiplyed by learning rate:%f\nshould be:\n%v\ngot:\n%v", delta, bias, test.learningRate, expected, nn.outputLayer.Bias)
		}
	}
}

func TestUpdateBiasOutFail(t *testing.T) {
	for _, test := range updateBiasOutTestFail {
		goErr := make(chan error)
		delta := matrix.NewFrom(test.delta)
		bias := matrix.NewFrom(test.biasWeights)

		nn, _ := New(test.hiddenLayers, test.hiddenLayers[len(test.hiddenLayers)-1], test.actFn, "relu", 1, 1, test.learningRate)
		nn.deltas[len(nn.deltas)-1] = delta
		nn.outputLayer.Bias = bias

		go nn.updateBias(len(nn.deltas)-1, goErr)

		err, _ := <-goErr
		if err == nil {
			t.Errorf("supposed to fail here!")
		}
	}
}

func TestUpdateWeights(t *testing.T) {
	for _, test := range updateWeightTest {
		goErr := make(chan error)
		index := test.index
		delta := matrix.NewFrom(test.delta)
		weight := matrix.NewFrom(test.biasWeights)
		expected := matrix.NewFrom(test.expected)

		nn, _ := New(test.hiddenLayers, 1, test.actFn, "relu", 1, 1, test.learningRate)
		nn.weights[index] = delta
		nn.hiddenLayers[index].Weights = weight

		go nn.updateWeight(index, goErr)

		err, _ := <-goErr
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		if !expected.Equal(nn.hiddenLayers[index].Weights) {
			t.Errorf("\n\nweight:\n%v\ndelta weights:\n%v\nmultiplyed by learning rate:%f\nshould be:\n%v\ngot:\n%v", delta, weight, test.learningRate, expected, nn.hiddenLayers[index].Weights)
		}
	}
}
