package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
)

var (
	config       = flag.String("config", "", "load a json file with all neural network configurations")
	train        = flag.Bool("train", false, "train a model")
	predict      = flag.Bool("predict", false, "predict loading model")
	model        = flag.String("model", "", "model name to load or save")
	learningRate = flag.Float64("lr", 0, "learning rate (between 0 and 1)")
	hidden       = flag.String("hidden", "", "give your hidden layers in and out size with comma seperated values like so: -hidden 3,4,5,2")
	output       = flag.Int("output", 0, "output vector size")
	actFn        = flag.String("hActFn", "", "hidden layers activation function")
	outActFn     = flag.String("oActFn", "", "output layer activation function")
	batch        = flag.Int("batch", 0, "batch size")
	epochs       = flag.Int("epochs", 0, "number of epochs")
	trainData    = flag.String("trainData", "", "path to training data")
	valData      = flag.String("valData", "", "path to validation data")
	reader       = flag.String("reader", "", "path to dataset reader")
	file         = flag.String("file", "", "file to predict")
)

func main() {
	flag.Usage = func() {
		fmt.Printf("Usage of %v\n", os.Args[0])
		fmt.Println("\tYou can either train or predict a model using the fully connected")
		fmt.Println("\tneural network.")
		fmt.Println()
		fmt.Println("\tTo train load a json config file like in examples/mnist_config.json")
		fmt.Println("\tOr start the script with all needed flags:")
		fmt.Printf("\t\t%v -train -config [filename]\n", os.Args[0])
		fmt.Printf("\t\t%v -train -hidden 3,4,5 -output 10 -lr 0.1 ...\n", os.Args[0])
		fmt.Println()
		fmt.Println("\tTo predict load the model by name and a file")
		fmt.Printf("\t\t%v -predict -model [modelname] -file [filename]\n", os.Args[0])
		fmt.Println()
		flag.PrintDefaults()
	}
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n\n", err)
		flag.Usage()
		os.Exit(1)
	}
}

func run() error {
	flag.Parse()

	if *predict && *train {
		return errors.New("can't predict and train at the same time")
	} else if *predict && *model == "" {
		return errors.New("need a named model to load")
	} else if *predict && *model != "" {
		if _, err := loadModel(*model); err != nil {
			return err
		}
	} else if !*train {
		return errors.New("the neural network needs to do something. either train or predict!")
	} else if *config != "" {

		config, err := parseConfig(*config)
		if err != nil {
			return err
		}
		if err := start(config); err != nil {
			return err
		}
	}

	if *batch <= 0 ||
		*learningRate <= 0 ||
		*learningRate >= 1 ||
		*hidden == "" ||
		*output <= 0 ||
		*actFn == "" ||
		*outActFn == "" ||
		*epochs <= 0 ||
		*trainData == "" ||
		*valData == "" ||
		*reader == "" {
		return errors.New("either provide a config file or all flags")
	}

	// parse hiddenLayers
	tmp := strings.Split(*hidden, ",")
	hiddenLayers := []int{}
	for _, val := range tmp {
		valInt, err := strconv.Atoi(val)
		if err != nil {
			return err
		}
		hiddenLayers = append(hiddenLayers, valInt)
	}

	if err := start(Config{
		HiddenLayers:   hiddenLayers,
		LearningRate:   *learningRate,
		BatchSize:      *batch,
		Epochs:         *epochs,
		Output:         *output,
		Model:          *model,
		ActFn:          *actFn,
		OutActFn:       *outActFn,
		TrainData:      *trainData,
		ValidationData: *valData,
		Reader:         *reader,
	}); err != nil {
		return err
	}

	return nil
}
