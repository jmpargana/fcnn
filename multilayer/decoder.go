package multilayer

import (
	"bytes"
	"encoding/gob"
)

// MarshalBinary is the method needed to implement the BinaryMarshaler in the
// encoding package. It will be called to parse the mlp data into binary form
// to be saved in a file for later use.
func (m *MultiLayerPerceptron) MarshalBinary() ([]byte, error) {
	w := wrapMultiLayerPerceptron{
		HiddenLayers: m.HiddenLayers,
		OutputLayer:  m.outputLayer,
		BatchSize:    m.batchSize,
		EpochSize:    m.epochSize,
		LearningRate: m.learningRate,
		Weights:      m.weights,
		Deltas:       m.deltas,
		LastInput:    m.lastInput,
		Reader:       m.Reader,
	}

	buf := new(bytes.Buffer)
	if err := gob.NewEncoder(buf).Encode(&w); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// UnmarshalBinary is the method needed to implement the BinaryUnmarshaler in the
// encoding package. It will be called to parse a gob file and assign the data to
// and instance of an mlp.
func (m *MultiLayerPerceptron) UnmarshalBinary(data []byte) error {
	w := wrapMultiLayerPerceptron{}

	reader := bytes.NewReader(data)
	if err := gob.NewDecoder(reader).Decode(&w); err != nil {
		return err
	}

	m.HiddenLayers = w.HiddenLayers
	m.outputLayer = w.OutputLayer
	m.batchSize = w.BatchSize
	m.epochSize = w.EpochSize
	m.learningRate = w.LearningRate
	m.weights = w.Weights
	m.deltas = w.Deltas
	m.lastInput = w.LastInput
	m.Reader = w.Reader

	return nil
}
