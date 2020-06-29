package readers

import "github.com/jmpargana/matrix"

// Instance is a struct that contains the input and expected output in form of
// two vectors.
type Instance struct {
	Image, Label matrix.Matrix
}

// Reader interface is satisfied with Read method which given two filepaths
// returns a slice of Instances.
type Reader interface {
	// TODO: load train and test data seperately
	Read(train, test string) ([]Instance, error)
	// DataFromFile is used to perform a predition. It shouldn't have any information
	// about the expected output so it simply parses a vector from an png image.
	DataFromFile(filename string) (matrix.Matrix, error)
}

var DatasetReaders = map[string]Reader{
	"mnist": Mnist{},
}
