package fcnn

import (
	// "encoding/json"
	"errors"
	"os"

	"github.com/jmpargana/fcnn/layer"
	"github.com/jmpargana/matrix"
)

func New(hiddenLayers []int,
	outputLayer int,
	hiddenActFn,
	outputActFn string,
	batchSize,
	epochSize int,
	learningRate float64) (MultiLayerPerceptron, error) {

	if batchSize < 1 || epochSize < 1 {
		return MultiLayerPerceptron{}, errors.New("can't create neural network with negative batch or epoch size")
	}

	if learningRate <= 0 || learningRate >= 1 {
		return MultiLayerPerceptron{}, errors.New("learning rate needs to have value between 0 and 1")
	}

	hLayers, err := startHiddenLayers(hiddenActFn, hiddenLayers)
	if err != nil {
		return MultiLayerPerceptron{}, err
	}

	oLayer, err := layer.New(outputActFn, hiddenLayers[len(hiddenLayers)-1], outputLayer)
	if err != nil {
		return MultiLayerPerceptron{}, err
	}

	activations := make([]matrix.Matrix, len(hiddenLayers))
	weights := make([]matrix.Matrix, len(hiddenLayers))
	deltas := make([]matrix.Matrix, len(hiddenLayers))

	return MultiLayerPerceptron{
		hiddenLayers: hLayers,
		outputLayer:  oLayer,
		batchSize:    batchSize,
		epochSize:    epochSize,
		learningRate: learningRate,
		activations:  activations,
		weights:      weights,
		deltas:       deltas,
	}, nil
}

// startHiddeLayers is a helper function to seperate concerns and make it
// easier to unittest every single part of the code that might fail.
func startHiddenLayers(hiddenActFn string, hiddenLayers []int) ([]layer.Layer, error) {
	if len(hiddenLayers) < 1 {
		return nil, errors.New("at least one hidden layer is needed")
	}

	hLayers := make([]layer.Layer, 0, len(hiddenLayers)-1)

	for i := 0; i < len(hiddenLayers)-1; i++ {
		l, err := layer.New(hiddenActFn, hiddenLayers[i], hiddenLayers[i+1])
		if err != nil {
			return nil, err
		}

		hLayers = append(hLayers, l)
	}

	return hLayers, nil
}

// TODO: implement or abstract. This configuration could be read in the main function
// from some json file and the New constructor gets called.
func NewConfig(config os.File) (MultiLayerPerceptron, error) {
	return MultiLayerPerceptron{}, nil
}
