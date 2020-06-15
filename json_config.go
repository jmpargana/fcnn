package fcnn

import (
	"encoding/json"
	"io/ioutil"
)

type Config struct {
	LearningRate   float64 `json:"learning_rate"`
	HiddenLayers   []int   `json:"hidden_layers"`
	Output         int     `json:"output"`
	Epochs         int     `json:"epochs"`
	BatchSize      int     `json:"batch_size"`
	Reader         string  `json:"reader"`
	TrainData      string  `json:"train_data"`
	ValidationData string  `json:"validation_data"`
}

func parseConfig(filename string) (Config, error) {

	file, err := ioutil.ReadFile(filename)
	if err != nil {
		return Config{}, err
	}

	config := Config{}

	if err := json.Unmarshal(file, &config); err != nil {
		return Config{}, err
	}

	return config, nil
}
