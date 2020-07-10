package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"time"

	"github.com/cheggaaa/pb"
	"github.com/jmpargana/fcnn/multilayer"
	"github.com/jmpargana/fcnn/readers"
)

func start(conf Config) error {
	nn, err := fromParsedConfig(conf)
	if err != nil {
		return err
	}

	train, err := readers.DatasetReaders[nn.Reader].DataFrom(conf.TrainData, conf.ValidationData)
	if err != nil {
		return fmt.Errorf("failed to read training data: %v", err)
	}

	for i := 0; i < conf.Epochs; i++ {
		trainBar := pb.StartNew(len(train))
		log.Printf("Starting epoch nr: %d\n", i+1)

		for j := range train {
			trainBar.Increment()

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
		trainBar.Finish()
	}
	// TODO: validate nn with test images

	return saveNetwork(nn, conf.Model, conf.Reader)
}

func runPrediction(modelName, filename string) error {
	nn, err := loadModel(modelName)
	if err != nil {
		return err
	}
	fmt.Println(nn.String())

	result, err := readers.DatasetReaders[nn.Reader].PredictDataFrom(filename)
	fmt.Println(result)

	return err
}

// loadModel loads a gob file with an existing trained neural network.
func loadModel(modelName string) (multilayer.MultiLayerPerceptron, error) {
	f, err := os.Open("models/" + modelName + ".model.gob")
	if err != nil {
		return multilayer.MultiLayerPerceptron{}, fmt.Errorf("failed opening model: %v", err)
	}
	defer f.Close()

	if err != nil {
		return multilayer.MultiLayerPerceptron{}, err
	}

	nn := multilayer.MultiLayerPerceptron{}
	data, err := ioutil.ReadAll(f)
	if err != nil {
		return multilayer.MultiLayerPerceptron{}, err
	}
	err = nn.UnmarshalBinary(data)
	return nn, err
}

// saveNetwork uploads the binary encoded structure of an instance of the neural
// network.
func saveNetwork(nn multilayer.MultiLayerPerceptron, modelName, reader string) error {
	if modelName == "" {
		modelName = reader + time.Now().String()
	}

	log.Printf("Saving trained model: %s in models directory", modelName)
	_ = os.Mkdir("models/", os.ModePerm)

	data, err := nn.MarshalBinary()
	if err != nil {
		return err
	}
	return ioutil.WriteFile("models/"+modelName+".model.gob", data, 0644)
}

// fromParsedConfig just creates an instance of a fcnn given a Config struct.
// It will fail if any of the settings are invalid and return an error.
func fromParsedConfig(conf Config) (multilayer.MultiLayerPerceptron, error) {
	return multilayer.New(
		conf.HiddenLayers,
		conf.Output,
		conf.ActFn,
		conf.OutActFn,
		conf.BatchSize,
		conf.Epochs,
		conf.LearningRate,
		conf.Reader)
}
