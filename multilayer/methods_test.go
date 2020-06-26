package multilayer

import (
	"sync"
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

		nn, _ := New(test.HiddenLayers, 1, test.actFn, "relu", 1, 1, 0.5)
		nn.HiddenLayers[len(nn.HiddenLayers)-2].Sum = sum
		nn.HiddenLayers[len(nn.HiddenLayers)-1].Weights = matrixPlus1
		nn.deltas[len(nn.HiddenLayers)-1] = prevDelta

		if err := nn.calculateDelta(len(nn.HiddenLayers)-2, matrix.Matrix{}); err != nil {
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

		nn, _ := New(test.HiddenLayers, matrixPlus1.NumRows, test.actFn, "relu", 1, 1, 0.5)
		nn.HiddenLayers[len(nn.HiddenLayers)-1].Sum = sum
		nn.outputLayer.Weights = matrixPlus1
		nn.deltas[len(nn.HiddenLayers)] = prevDelta

		if err := nn.calculateDelta(len(nn.HiddenLayers)-1, matrix.Matrix{}); err != nil {
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

		nn, _ := New(test.HiddenLayers, 1, test.actFn, "relu", 1, 1, 0.5)
		nn.HiddenLayers[len(nn.HiddenLayers)-2].Sum = sum
		nn.HiddenLayers[len(nn.HiddenLayers)-1].Weights = matrixPlus1
		nn.deltas[len(nn.HiddenLayers)-1] = prevDelta

		if err := nn.calculateDelta(len(nn.HiddenLayers)-2, matrix.Matrix{}); err == nil {
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

		nn, _ := New(test.HiddenLayers, 1, test.actFn, "relu", 1, 1, test.learningRate)
		nn.deltas[index] = delta
		nn.HiddenLayers[index].Bias = bias

		wg := new(sync.WaitGroup)
		wg.Add(1)
		go nn.updateBias(index, goErr, wg)
		wg.Wait()

		err, _ := <-goErr
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		if !expected.Equal(nn.HiddenLayers[index].Bias) {
			t.Errorf("\nbias:\n%v\ndelta:\n%v\nmultiplyed by learning rate:%f\nshould be:\n%v\ngot:\n%v", delta, bias, test.learningRate, expected, nn.HiddenLayers[index].Bias)
		}
	}
}

func TestUpdateBiasOut(t *testing.T) {
	for _, test := range updateBiasOutTest {
		goErr := make(chan error)
		delta := matrix.NewFrom(test.delta)
		bias := matrix.NewFrom(test.biasWeights)
		expected := matrix.NewFrom(test.expected)

		nn, _ := New(test.HiddenLayers, test.HiddenLayers[len(test.HiddenLayers)-1], test.actFn, "relu", 1, 1, test.learningRate)
		nn.deltas[len(nn.deltas)-1] = delta
		nn.outputLayer.Bias = bias

		wg := new(sync.WaitGroup)
		wg.Add(1)
		go nn.updateBias(len(nn.deltas)-1, goErr, wg)
		wg.Wait()

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

		nn, _ := New(test.HiddenLayers, test.HiddenLayers[len(test.HiddenLayers)-1], test.actFn, "relu", 1, 1, test.learningRate)
		nn.deltas[len(nn.deltas)-1] = delta
		nn.outputLayer.Bias = bias

		wg := new(sync.WaitGroup)
		wg.Add(1)
		go nn.updateBias(len(nn.deltas)-1, goErr, wg)
		wg.Wait()

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

		nn, _ := New(test.HiddenLayers, 1, test.actFn, "relu", 1, 1, test.learningRate)
		nn.weights[index] = delta
		nn.HiddenLayers[index].Weights = weight

		wg := new(sync.WaitGroup)
		wg.Add(1)
		go nn.updateWeight(index, goErr, wg)
		wg.Wait()

		err, _ := <-goErr
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		if !expected.Equal(nn.HiddenLayers[index].Weights) {
			t.Errorf("\n\nweight:\n%v\ndelta weights:\n%v\nmultiplyed by learning rate:%f\nshould be:\n%v\ngot:\n%v", delta, weight, test.learningRate, expected, nn.HiddenLayers[index].Weights)
		}
	}
}

func TestUpdateWeightsOut(t *testing.T) {
	for _, test := range updateWeightTestOut {
		goErr := make(chan error)
		index := len(test.HiddenLayers) - 1
		delta := matrix.NewFrom(test.delta)
		weight := matrix.NewFrom(test.biasWeights)
		expected := matrix.NewFrom(test.expected)

		nn, _ := New(test.HiddenLayers, 1, test.actFn, "relu", 1, 1, test.learningRate)
		nn.weights[index] = delta
		nn.outputLayer.Weights = weight

		wg := new(sync.WaitGroup)
		wg.Add(1)
		go nn.updateWeight(index, goErr, wg)
		wg.Wait()

		err, _ := <-goErr
		if err != nil {
			t.Errorf("shouldn't fail here: %v", err)
		}

		if !expected.Equal(nn.outputLayer.Weights) {
			t.Errorf("\n\nweight:\n%v\ndelta weights:\n%v\nmultiplyed by learning rate:%f\nshould be:\n%v\ngot:\n%v", delta, weight, test.learningRate, expected, nn.outputLayer.Weights)
		}
	}
}

func TestUpdateWeightsFail(t *testing.T) {
	for _, test := range updateWeightTestFail {
		goErr := make(chan error)
		index := test.index
		delta := matrix.NewFrom(test.delta)
		weight := matrix.NewFrom(test.biasWeights)

		nn, _ := New(test.HiddenLayers, 1, test.actFn, "relu", 1, 1, test.learningRate)
		nn.weights[index] = delta
		nn.HiddenLayers[index].Weights = weight

		wg := new(sync.WaitGroup)
		wg.Add(1)
		go nn.updateWeight(index, goErr, wg)
		wg.Wait()

		err, _ := <-goErr
		if err == nil {
			t.Errorf("shouldn't fail here: %v", err)
		}
	}
}

func TestCalculateWeight(t *testing.T) {
	for _, test := range calculateWeightTest {
		nn, _ := New(test.HiddenLayers, 1, "relu", "relu", 1, 1, 0.1)
		index := test.index

		prevOut := matrix.NewFrom(test.prevOut)
		delta := matrix.NewFrom(test.delta)
		expected := matrix.NewFrom(test.expected)

		nn.deltas[index] = delta
		nn.HiddenLayers[index-1].Output = prevOut

		goErr := make(chan error)
		wg := new(sync.WaitGroup)
		wg.Add(1)
		go nn.calculateWeight(index, goErr, wg)

		err := <-goErr
		if err != nil {
			t.Errorf("wans't supposed to fail here: %v", err)
		}

		if !nn.weights[index].Equal(expected) {
			t.Errorf("\nexpected:\n%vgot:\n%v", expected, nn.weights[index])
		}
	}
}

func TestCalculateWeightFirst(t *testing.T) {
	for _, test := range calculateWeightTestOut {
		nn, _ := New(test.HiddenLayers, 1, "relu", "relu", 1, 1, 0.1)

		prevOut := matrix.NewFrom(test.prevOut)
		delta := matrix.NewFrom(test.delta)
		expected := matrix.NewFrom(test.expected)

		nn.deltas[0] = delta
		nn.lastInput = prevOut

		goErr := make(chan error)
		wg := new(sync.WaitGroup)
		wg.Add(1)
		go nn.calculateWeight(0, goErr, wg)

		err := <-goErr
		if err != nil {
			t.Errorf("wans't supposed to fail here: %v", err)
		}

		if !nn.weights[0].Equal(expected) {
			t.Errorf("\nexpected:\n%vgot:\n%v", expected, nn.weights[0])
		}
	}
}

func TestInvalidCalculateWeight(t *testing.T) {
	for _, test := range calculateWeightTestInvalid {
		nn, _ := New(test.HiddenLayers, 1, "relu", "relu", 1, 1, 0.1)

		prevOut := matrix.NewFrom(test.prevOut)
		delta := matrix.NewFrom(test.delta)
		index := test.index

		nn.deltas[index] = delta
		nn.HiddenLayers[index-1].Output = prevOut

		goErr := make(chan error)
		wg := new(sync.WaitGroup)
		wg.Add(1)
		go nn.calculateWeight(0, goErr, wg)

		err := <-goErr
		if err == nil {
			t.Errorf("supposed to fail here!")
		}
	}
}

func TestGradientDescent(t *testing.T) {
	for _, test := range gradientDescentTest {
		nn, _ := New(test.HiddenLayers, test.outputLayer, "relu", "relu", 1, 1, test.learningRate)

		var deltaBias []matrix.Matrix
		var deltaWeights []matrix.Matrix
		var expectedWeights []matrix.Matrix
		var expectedBias []matrix.Matrix

		// initialize slice of deltas (weights and bias)
		// and expected results
		for i := range test.deltaWeights {
			deltaBias = append(deltaBias, matrix.NewFrom(test.deltaBias[i]))
			deltaWeights = append(deltaWeights, matrix.NewFrom(test.deltaWeights[i]))
			expectedWeights = append(expectedWeights, matrix.NewFrom(test.expectedWeights[i]))
			expectedBias = append(expectedBias, matrix.NewFrom(test.expectedBias[i]))
		}

		// save current bias and weights in each layer
		for i := range nn.HiddenLayers {
			nn.HiddenLayers[i].Bias = matrix.NewFrom(test.bias[i])
			nn.HiddenLayers[i].Weights = matrix.NewFrom(test.weights[i])
		}
		nn.outputLayer.Bias = matrix.NewFrom(test.bias[len(nn.HiddenLayers)])
		nn.outputLayer.Weights = matrix.NewFrom(test.weights[len(nn.HiddenLayers)])

		nn.deltas = deltaBias
		nn.weights = deltaWeights

		if err := nn.GradientDescent(); err != nil {
			t.Errorf("failed with: %v", err)
		}

		// time.Sleep(1000000)

		for i, l := range nn.HiddenLayers {
			if !l.Bias.Equal(expectedBias[i]) {
				t.Errorf("\nexpected:\n%v\ngot:\n%v\n", expectedBias[i], l.Bias)
			}
			if !l.Weights.Equal(expectedWeights[i]) {
				t.Errorf("\nexpected:\n%v\ngot:\n%v\n", expectedWeights[i], l.Weights)
			}
		}

		lastIndex := len(nn.HiddenLayers)

		if !nn.outputLayer.Bias.Equal(expectedBias[lastIndex]) {
			t.Errorf("\nexpected:\n%v\ngot:\n%v\n", expectedBias[lastIndex], nn.outputLayer.Bias)
		}
		if !nn.outputLayer.Weights.Equal(expectedWeights[lastIndex]) {
			t.Errorf("\nexpected:\n%v\ngot:\n%v\n", expectedWeights[lastIndex], nn.outputLayer.Weights)
		}
	}
}
