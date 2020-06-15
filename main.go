package fcnn

import (
	"flag"
	"fmt"
)

var (
	config       = flag.String("config", "", "load a json file with all neural network configurations")
	train        = flag.Bool("train", false, "train a model")
	predict      = flag.Bool("predict", false, "predict loading model")
	model        = flag.String("model", "", "model name to load")
	learningRate = flag.Float64("lr", 0, "learning rate")
)

func main() {
	flag.Parse()

	flag.Usage = func() {
		fmt.Println("Usage")
		flag.PrintDefaults()
	}

}
