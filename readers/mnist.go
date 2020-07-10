package readers

import (
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/jmpargana/fcnn/readers/mnist"
	"github.com/jmpargana/matrix"
)

// Mnist reader with two methods that satisfy the Reader interface.
type Mnist struct{}

func (m Mnist) DataFrom(images, labels string) ([]Instance, error) {
	dataset, err := mnist.ReadDataSet(images, labels)
	if err != nil {
		return nil, fmt.Errorf("failed loading dataset: %v", err)
	}

	instances := make([]Instance, dataset.N)
	for i, dataImage := range dataset.Data {
		label := matrix.New(10, 1)
		if err := label.Set(dataImage.Digit, 0, 1); err != nil {
			return nil, fmt.Errorf("failed assigning label: %v", err)
		}

		instances[i] = Instance{
			Image: matrix.NewFrom(dataImage.Image),
			Label: label,
		}
	}

	return instances, nil
}

// Credits to: https://github.com/sausheong/gonn
func (m Mnist) PredictDataFrom(filename string) (matrix.Matrix, error) {
	f, err := os.Open(filename)
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
