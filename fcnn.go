package main

import (
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
		// TODO: log intermediate scores to user and
		// should only perform gradient descent batchwise
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
		}
	}

	if err := saveNetwork(nn, conf.Model, conf.Reader); err != nil {
		return err
	}
	return nil
}

func loadModel(model string) error {
	// h, err := os.Create("models/" + model + "_hidden_weights.model")
	// if err != nil {
	// 	return err
	// }
	// o, err := os.Create("models/" + model + "_output_weights.model")
	// if err != nil {
	// 	return err
	// }

	// nn, err := multilayer.New()

	return nil
}

func saveNetwork(nn multilayer.MultiLayerPerceptron, modelName, reader string) error {
	if modelName == "" {
		modelName = reader + time.Now().String()
	}

	// h, err := os.Create("models/" + modelName + "_hidden_weights.model")
	// if err != nil {
	// 	return err
	// }
	// o, err := os.Create("models/" + modelName + "_output_weights.model")
	// if err != nil {
	// 	return err
	// }

	// // TODO: method not available for matrix, needs to be implemented from scratch
	// nn.HiddenLayers.MarshalBinaryTo(h)
	// nn.Output.MarshalBinaryTo(o)

	// h.Close()
	// o.Close()
	return nil
}
