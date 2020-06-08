package fcnn

import "github.com/jmpargana/fcnn/layer"

// MultiLayerPerceptron is an abstraction of the hiddenLayers and outputLayer.
// It performs the forward propagation sequentially layer by layer for each epoch,
// but concurrently for all vectors in the expected batch size.
type MultiLayerPerceptron struct {
	hiddenLayers         []layer.Layer
	outputLayer          layer.Layer
	batchSize, epochSize int
}
