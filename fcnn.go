package main

import (
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

	// TODO: save model when done

	return nil
}

func loadModel(model string) error {
	return nil
}
