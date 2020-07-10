package main

import (
	"encoding/json"
	"io/ioutil"
)

type Config struct {
	LearningRate   float64 `json:"learning_rate"`
	HiddenLayers   []int   `json:"hidden_layers"`
	Output         int     `json:"output"`
	ActFn          string  `json:"activation_function"`
	OutActFn       string  `json:"out_activation_function"`
	Model          string  `json:"model"`
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

	err = json.Unmarshal(file, &config)
	return config, err
}

func (c Config) equal(other Config) bool {
	for i := range c.HiddenLayers {
		if c.HiddenLayers[i] != other.HiddenLayers[i] {
			return false
		}
	}
	return len(c.HiddenLayers) == len(other.HiddenLayers) &&
		c.TrainData == other.TrainData &&
		c.ValidationData == other.ValidationData &&
		c.BatchSize == other.BatchSize &&
		c.Epochs == other.Epochs &&
		c.Output == other.Output &&
		c.Reader == other.Reader
}
