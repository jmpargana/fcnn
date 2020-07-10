package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
)

var (
	train   = flag.Bool("train", false, "train a model")
	predict = flag.Bool("predict", false, "predict by loading model")
	config  = flag.String("config", "", "load a json file with all neural network configurations")
	model   = flag.String("model", "", "model name to load or save")
	file    = flag.String("file", "", "file to predict")
)

func main() {
	flag.Usage = func() {
		fmt.Printf("Usage of %v\n", os.Args[0])
		fmt.Println("\tYou can either train a model or predict using a fcnn")
		fmt.Println()
		fmt.Println("\tTo train load a json config file like any one in the examples dir.")
		fmt.Printf("\t\t%v -train -config [filename]\n", os.Args[0])
		fmt.Println()
		fmt.Println("\tTo predict load the model by name and a file. By default all models")
		fmt.Println("\tare saved in the models folder with an .model.gob extension.")
		fmt.Printf("\t\t%v -predict -model [modelname] -file [filename]\n", os.Args[0])
		fmt.Println()
		flag.PrintDefaults()
	}
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "\n%v\n", err)
		os.Exit(1)
	}
}

func run() error {
	flag.Parse()

	if *predict && *train {
		flag.Usage()
		return errors.New("can't predict and train at the same time")
	} else if !*predict && !*train {
		flag.Usage()
		return errors.New("the neural network needs to do something. either train or predict!")
	} else if *predict && (*model == "" || *file == "") {
		flag.Usage()
		return errors.New("need a named model to load and file to predict")
	} else if *predict && *model != "" && *file != "" {

		return runPrediction(*model, *file)

	} else if *config == "" {
		flag.Usage()
		return errors.New("need a config file to load")
	}

	config, err := parseConfig(*config)
	if err != nil {
		return fmt.Errorf("failed to parse config file: %v", err)
	}

	return start(config)
}
