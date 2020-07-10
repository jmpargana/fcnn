package main

import "testing"

func TestParseConfig(t *testing.T) {
	tt := []struct {
		name, filename string
		expected       Config
	}{
		{
			name:     "reading available file",
			filename: "./examples/mnist_config.json",
			expected: Config{
				LearningRate:   0.1,
				HiddenLayers:   []int{784, 100, 50},
				Output:         10,
				ActFn:          "relu",
				OutActFn:       "softmax",
				Model:          "mnist",
				BatchSize:      10,
				Epochs:         3,
				Reader:         "mnist",
				TrainData:      "datasets/train-images-idx3-ubyte",
				ValidationData: "datasets/train-labels-idx1-ubyte",
			},
		},
	}

	for _, test := range tt {
		t.Run(test.name, func(t *testing.T) {
			got, err := parseConfig(test.filename)
			if err != nil {
				t.Errorf("failed with: %v", err)
			}

			if !got.equal(test.expected) {
				t.Errorf("got: %v; expected %v", got, test.expected)
			}
		})
	}
}

func TestParseConfigDifferent(t *testing.T) {
	tt := []struct {
		name, filename string
		expected       Config
	}{
		{
			name:     "unequal file content",
			filename: "./examples/mnist_config.json",
			expected: Config{
				LearningRate:   0.1,
				HiddenLayers:   []int{784, 100, 30},
				Output:         10,
				BatchSize:      10,
				Epochs:         3,
				Reader:         "../readers/mnist.go",
				TrainData:      "../datasets/train-images-idx3-ubyte",
				ValidationData: "../datasets/train-labels-idx3-ubyte",
			},
		},
	}

	for _, test := range tt {
		t.Run(test.name, func(t *testing.T) {
			got, err := parseConfig(test.filename)
			if err != nil {
				t.Errorf("failed with: %v", err)
			}

			if got.equal(test.expected) {
				t.Errorf("got: %v; expected %v", got, test.expected)
			}
		})
	}
}

func TestParseConfigFail(t *testing.T) {
	tt := []struct {
		name, filename string
		expected       Config
	}{
		{
			name:     "unavailable file",
			filename: "./examples/non.json",
		},
	}

	for _, test := range tt {
		t.Run(test.name, func(t *testing.T) {
			if _, err := parseConfig(test.filename); err == nil {
				t.Errorf("should have failed here")
			}
		})
	}
}

func TestParseConfigFail2(t *testing.T) {
	tt := []struct {
		name, filename string
		expected       Config
	}{
		{
			name:     "non parsable json",
			filename: "./main.go",
		},
	}

	for _, test := range tt {
		t.Run(test.name, func(t *testing.T) {
			if _, err := parseConfig(test.filename); err == nil {
				t.Errorf("should have failed here")
			}
		})
	}
}
