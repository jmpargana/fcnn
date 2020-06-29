package multilayer

import (
	"github.com/jmpargana/fcnn/layer"
	"github.com/jmpargana/matrix"
)

// MultiLayerPerceptron is an abstraction of the HiddenLayers and outputLayer.
// It performs the forward propagation sequentially layer by layer for each epoch,
// but concurrently for all vectors in the expected batch size.
type MultiLayerPerceptron struct {
	HiddenLayers         []layer.Layer
	outputLayer          layer.Layer
	batchSize, epochSize int
	learningRate         float64
	deltas, weights      []matrix.Matrix
	lastInput            matrix.Matrix
	Reader               string
}

// wrapMultiLayerPerceptron is a wrapper to help marshaling the neural network
// with the gob encoder/decoder.
type wrapMultiLayerPerceptron struct {
	HiddenLayers         []layer.Layer
	OutputLayer          layer.Layer
	BatchSize, EpochSize int
	LearningRate         float64
	Deltas, Weights      []matrix.Matrix
	LastInput            matrix.Matrix
	Reader               string
}
