package multilayer

import (
	// "encoding/json"
	"errors"

	"github.com/jmpargana/fcnn/layer"
	"github.com/jmpargana/matrix"
)

// New instantiates a Fully Connected Neural Network or MultiLayerPerceptron.
// It receives a slice of in/out sizes for each hidden layer and an out size
// for the output layer. So ([1, 2, 3], 4) would mean 2 hidden layers and 1
// in/output with the respective sizes: 1->2, 2->3, 3->4.
// New also receives the activation functions for the hidden and output layers
// as well as a batch, epoch size and a learning rate.
func New(HiddenLayers []int,
	outputLayer int,
	hiddenActFn,
	outputActFn string,
	batchSize,
	epochSize int,
	learningRate float64,
	reader string) (MultiLayerPerceptron, error) {

	if batchSize < 1 || epochSize < 1 {
		return MultiLayerPerceptron{}, errors.New("can't create neural network with negative batch or epoch size")
	}

	if learningRate <= 0 || learningRate >= 1 {
		return MultiLayerPerceptron{}, errors.New("learning rate needs to have value between 0 and 1")
	}

	hLayers, err := startHiddenLayers(hiddenActFn, HiddenLayers)
	if err != nil {
		return MultiLayerPerceptron{}, err
	}

	oLayer, err := layer.New(outputActFn, HiddenLayers[len(HiddenLayers)-1], outputLayer)
	if err != nil {
		return MultiLayerPerceptron{}, err
	}

	// since the hidden layers already contains in and out sizes it has the
	// same length as counting with the output layer
	weights := make([]matrix.Matrix, len(HiddenLayers))
	deltas := make([]matrix.Matrix, len(HiddenLayers))
	lastInput := matrix.New(HiddenLayers[0], 1)

	return MultiLayerPerceptron{
		HiddenLayers: hLayers,
		outputLayer:  oLayer,
		batchSize:    batchSize,
		epochSize:    epochSize,
		learningRate: learningRate,
		weights:      weights,
		deltas:       deltas,
		lastInput:    lastInput,
		Reader:       reader,
	}, nil
}

// startHiddeLayers is a helper function to seperate concerns and make it
// easier to unittest every single part of the code that might fail.
func startHiddenLayers(hiddenActFn string, HiddenLayers []int) ([]layer.Layer, error) {
	if len(HiddenLayers) < 2 {
		return nil, errors.New("at least one hidden layer is needed, only input size provided")
	}

	hLayers := make([]layer.Layer, 0, len(HiddenLayers)-1)

	for i := 0; i < len(HiddenLayers)-1; i++ {
		l, err := layer.New(hiddenActFn, HiddenLayers[i], HiddenLayers[i+1])
		if err != nil {
			return nil, err
		}

		hLayers = append(hLayers, l)
	}

	return hLayers, nil
}
