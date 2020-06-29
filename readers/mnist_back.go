package readers

import (
	"bufio"
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/jmpargana/fcnn/readers/mnist"
	"github.com/jmpargana/matrix"
)

type Mnist struct{}

func (m Mnist) Read(train, test string) ([]Instance, error) {
	trainFile, err := os.Open(train)
	if err != nil {
		return nil, err
	}
	testFile, err := os.Open(test)
	if err != nil {
		return nil, err
	}

	trainScanner := bufio.NewScanner(trainFile)
	testScanner := bufio.NewScanner(testFile)

	// just space holder so package compiles
	_, _ = trainScanner, testScanner

	return []Instance{}, nil
}

// Credits to: https://github.com/sausheong/gonn
func (m Mnist) DataFromFile(filename string) (matrix.Matrix, error) {
	f, err := os.Open("../" + filename)
	defer f.Close()
	if err != nil {
		return matrix.Matrix{}, err
	}

	img, err := png.Decode(f)
	if err != nil {
		return matrix.Matrix{}, err
	}

	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			gray.Set(x, y, img.At(x, y))
		}
	}

	pixels := make([]float64, len(gray.Pix))

	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.99) + 0.01
	}

	return matrix.NewFromVec(len(pixels), 1, pixels), nil
}

func printData(dataSet *mnist.DataSet, index int) {
	data := dataSet.Data[index]
	fmt.Println(data.Digit)
	mnist.PrintImage(data.Image)
}

func tester() {
	dataSet, err := mnist.ReadTrainSet("./mnist")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(dataSet.N)
	fmt.Println(dataSet.W)
	fmt.Println(dataSet.H)

	printData(dataSet, 0)
}
