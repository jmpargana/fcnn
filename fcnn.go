package main

import (
	"io/ioutil"
	"os"
	"time"

	"github.com/jmpargana/fcnn/multilayer"
	"github.com/jmpargana/fcnn/readers"
)

func start(conf Config) error {
	// load data from reader using a bufio not to overload cpu memory
	train, err := readers.Mnist(conf.TrainData, conf.ValidationData)

	nn, err := multilayer.New(
		conf.HiddenLayers,
		conf.Output,
		conf.ActFn,
		conf.OutActFn,
		conf.BatchSize,
		conf.Epochs,
		conf.LearningRate)
	if err != nil {
		return err
	}

	for i := 0; i < conf.Epochs; i++ {
		for j := range train {
			if _, err := nn.ForwProp(train[j].Image); err != nil {
				return err
			}
			if err := nn.BackProp(train[j].Label); err != nil {
				return err
			}
			if err := nn.GradientDescent(); err != nil {
				return err
			}
			// TODO: log intermediate scores to user and
			// should only perform gradient descent batchwise
		}
	}

	if err := saveNetwork(nn, conf.Model, conf.Reader); err != nil {
		return err
	}
	return nil
}

// loadModel loads a gob file with an existing trained neural network.
func loadModel(model string) (multilayer.MultiLayerPerceptron, error) {
	f, err := os.Open(model)
	if err != nil {
		return multilayer.MultiLayerPerceptron{}, err
	}

	nn := multilayer.MultiLayerPerceptron{}
	data, err := ioutil.ReadAll(f)
	if err != nil {
		return multilayer.MultiLayerPerceptron{}, err
	}

	if err := nn.UnmarshalBinary(data); err != nil {
		return multilayer.MultiLayerPerceptron{}, err
	}

	return nn, nil
}

// saveNetwork uploads the binary encoded structure of an instance of the neural
// network.
func saveNetwork(nn multilayer.MultiLayerPerceptron, modelName, reader string) error {
	if modelName == "" {
		modelName = reader + time.Now().String()
	}

	data, err := nn.MarshalBinary()
	if err != nil {
		return err
	}

	if err := ioutil.WriteFile("models/"+modelName+".model.gob", data, 0644); err != nil {
		return err
	}

	return nil
}
