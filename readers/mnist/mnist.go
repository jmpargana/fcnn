// MNIST database reader
// http://yann.lecun.com/exdb/mnist/
// https://gist.githubusercontent.com/higuma/dbcd006546eb844c01e5102b4d0bcc93/raw/e3f3b72a0d3a212bfa0b65feb52d7a73bcf72913/mnist.go
package mnist

import (
	"fmt"
	"os"
)

const (
	labelsFileMagic = 0x00000801
	imagesFileMagic = 0x00000803

	msgInvalidFormat = "Invalid format: %s"
	msgSizeUnmatch   = "Data size does not match: %s %s"
)

func fileError(f *os.File) error {
	return fmt.Errorf(msgInvalidFormat, f.Name())
}

// internal: Read 4 bytes and convert to big endian integer
func readInt32(f *os.File) (int, error) {
	buf := make([]byte, 4)
	n, e := f.Read(buf)
	switch {
	case e != nil:
		return 0, e
	case n != 4:
		return 0, fileError(f)
	}
	v := 0
	for _, x := range buf {
		v = v*256 + int(x)
	}
	return v, nil
}

// internal: raw image data
type imageData struct {
	N    int
	W    int
	H    int
	Data []uint8
}

func readImagesFile(path string) (*imageData, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	magic, e := readInt32(f)
	if e != nil || magic != imagesFileMagic {
		return nil, fileError(f)
	}
	n, e := readInt32(f)
	if e != nil {
		return nil, fileError(f)
	}
	w, e := readInt32(f)
	if e != nil {
		return nil, fileError(f)
	}
	h, e := readInt32(f)
	if e != nil {
		return nil, fileError(f)
	}
	sz := n * w * h
	data := &imageData{n, w, h, make([]uint8, sz)}
	len, e := f.Read(data.Data)
	if e != nil || len != sz {
		return nil, fileError(f)
	}
	return data, nil
}

// internal: raw label data
type labelData struct {
	N    int
	Data []uint8
}

func readLabelsFile(path string) (*labelData, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	magic, e := readInt32(f)
	if e != nil || magic != labelsFileMagic {
		return nil, fileError(f)
	}
	n, e := readInt32(f)
	if e != nil {
		return nil, fileError(f)
	}
	data := &labelData{n, make([]uint8, n)}
	len, e := f.Read(data.Data)
	if e != nil || len != n {
		return nil, fileError(f)
	}
	return data, nil
}

// Single digit+image datum
type DigitImage struct {
	Digit int
	Image [][]float64
}

// Data set
type DataSet struct {
	N    int
	W    int
	H    int
	Data []DigitImage
}

// Database readers
func ReadDataSet(imagesPath, labelsPath string) (*DataSet, error) {
	images, e := readImagesFile(imagesPath)
	if e != nil {
		return nil, e
	}
	labels, e := readLabelsFile(labelsPath)
	if e != nil {
		return nil, e
	}
	if images.N != images.N {
		return nil, fmt.Errorf(msgSizeUnmatch, labelsPath, imagesPath)
	}
	dataSet := &DataSet{N: images.N, W: images.W, H: images.H}
	dataSet.Data = make([]DigitImage, dataSet.N)
	rows := splitToRows(images.Data, images.N, images.H)
	for i := 0; i < dataSet.N; i++ {
		data := &dataSet.Data[i]
		data.Digit = int(labels.Data[i])
		data.Image = rows[0:dataSet.H]
		rows = rows[dataSet.H:]
	}
	return dataSet, nil
}

func splitToRows(data []uint8, N, H int) [][]float64 {
	nR := N * H
	rows := make([][]float64, nR)
	for i := 0; i < nR; i++ {
		floatData := make([]float64, H)
		for j := 0; j < H; j++ {
			floatData[j] = float64(data[j])
		}
		rows[i] = floatData[0:H]
		data = data[H:]
	}
	return rows
}
