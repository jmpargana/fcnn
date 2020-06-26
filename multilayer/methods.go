package multilayer

import (
	"bytes"
	"encoding/gob"
	"sync"

	"github.com/jmpargana/matrix"
)

// ForwProp calls the ForwProp method for each layer sequentially saving its output
// in the in vector each time.
// It returns an output vector which gets compared with the expected value.
func (m *MultiLayerPerceptron) ForwProp(in matrix.Matrix) (matrix.Matrix, error) {

	m.lastInput = in // this will be needed to complete the back propagation

	for _, layer := range m.HiddenLayers {

		var err error
		in, err = layer.ForwProp(in)
		if err != nil {
			return matrix.Matrix{}, err
		}
	}
	in, _ = m.outputLayer.ForwProp(in) // no way the size can't match by this point

	return in, nil
}

// BackProp receives an expected output Y, calculates the first delta
// using the quadratic function (might generalize later).
// It then propagates the error further saving the weight and bias errors
// indexing in the member attributes for later use with the Gradient Descent.
func (m *MultiLayerPerceptron) BackProp(output matrix.Matrix) error {

	goroutineErr := make(chan error)
	wgBackPropDone := make(chan struct{})
	wgBackProp := new(sync.WaitGroup)

	for i := len(m.deltas) - 1; i > 1; i-- {
		if err := m.calculateDelta(i, output); err != nil {
			return err
		}
		wgBackProp.Add(1)
		go m.calculateWeight(i, goroutineErr, wgBackProp)
	}

	go func() {
		wgBackProp.Wait()
		close(wgBackPropDone)
	}()

	select {
	case <-wgBackPropDone:
		close(goroutineErr) // close just to be safe
		break
	case err := <-goroutineErr:
		if err != nil {
			close(goroutineErr)
			return err
		}
	}
	return nil
}

// calculateDelta receives an error vector and calculates the error of each
// layer. Uses the quadratic function for the outputLayer and normal weight
// transposing for all others.
func (m *MultiLayerPerceptron) calculateDelta(index int, output matrix.Matrix) error {

	var delta matrix.Matrix
	var err error

	// assign delta an err according to which layer. last layer calculates quadratic error
	// last hidden layer uses the weights from the output layer
	// all others use the weights from the next layer.
	if index == len(m.deltas)-1 {
		delta, err = m.outputLayer.BackPropOutLayer(output)
	} else if index == len(m.deltas)-2 {
		delta, err = m.HiddenLayers[index].BackProp(
			m.deltas[index+1],
			m.outputLayer.Weights)
	} else {
		delta, err = m.HiddenLayers[index].BackProp(
			m.deltas[index+1],
			m.HiddenLayers[index+1].Weights)
	}

	if err != nil {
		return err
	}
	m.deltas[index] = delta

	return nil
}

// calculateWeight multiplies the previous activatedOutput with the current error
// generating a matrix of weights' errors.
func (m *MultiLayerPerceptron) calculateWeight(index int, goroutineErr chan error, wgBackProp *sync.WaitGroup) {
	var lastInput matrix.Matrix

	// this means the first layer, so we don't need the activation from the prev.
	// instead we use the input used for the feedforward
	if index == 0 {
		lastInput = m.lastInput
	} else {
		lastInput = m.HiddenLayers[index-1].Output
	}
	transposedPrevAct, _ := matrix.Trans(lastInput)

	weight, err := matrix.Mult(m.deltas[index], transposedPrevAct)
	if err != nil {
		goroutineErr <- err
	}
	m.weights[index] = weight

	goroutineErr <- nil
	wgBackProp.Done()
}

// GradientDescent is called after calculating the errors for both the bias and
// weights in each layer. It then sequentially updates both. It does so calling
// the method in each layer which simply subtracts the multiplyed error and learning
// rate from the current value.
func (m *MultiLayerPerceptron) GradientDescent() error {

	goErrs := make(chan error) // the updating can ran concurrently, we just need to check for errors
	wg := new(sync.WaitGroup)

	for i := 0; i <= len(m.HiddenLayers); i++ {
		wg.Add(2)
		go m.updateWeight(i, goErrs, wg)
		go m.updateBias(i, goErrs, wg)
	}

	wg.Wait()

	return nil
}

// updateWeights is ran concurrently and updates the weights from either the output layer
// or one of the hidden ones by calling the method of the type with the error that needs to
// be subtracted.
func (m *MultiLayerPerceptron) updateWeight(index int, goErr chan error, wg *sync.WaitGroup) {
	m.weights[index].MultScalar(m.learningRate)
	var err error

	if index == len(m.deltas)-1 {
		err = m.outputLayer.UpdateWeights(m.weights[index])
	} else {
		err = m.HiddenLayers[index].UpdateWeights(m.weights[index])
	}

	wg.Done()
	goErr <- err
}

// updateBias is ran concurrently and updates the weights from either the output layer
// or one of the hidden ones by calling the method of the type with the error that needs to
// be subtracted.
func (m *MultiLayerPerceptron) updateBias(index int, goErr chan error, wg *sync.WaitGroup) {

	m.deltas[index].MultScalar(m.learningRate)
	var err error

	if index == len(m.deltas)-1 {
		err = m.outputLayer.UpdateBias(m.deltas[index])
	} else {
		err = m.HiddenLayers[index].UpdateBias(m.deltas[index])
	}

	wg.Done()
	goErr <- err
}

func (m *MultiLayerPerceptron) MashalBinary() ([]byte, error) {
	b := new(bytes.Buffer)
	enc := gob.NewEncoder(b)
	if err := enc.Encode(m); err != nil {
		return nil, err
	}

	return b.Bytes(), nil
}

func (m *MultiLayerPerceptron) UnmarshalBinary(data []byte) error {
	mlp := new(MultiLayerPerceptron)
	b := new(bytes.Buffer)
	dec := gob.NewDecoder(b)
	if err := dec.Decode(mlp); err != nil {
		return err
	}
	m = mlp
	return nil
}
