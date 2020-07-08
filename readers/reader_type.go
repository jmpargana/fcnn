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
	// DataFrom fetches the data from the two seperate files which contain
	// both the images and labels. This procedure is equivalent when fetching
	// the training and test data.
	DataFrom(images, labels string) ([]Instance, error)
	// DataFromFile is used to perform a predition. It shouldn't have any information
	// about the expected output so it simply parses a vector from an png image.
	PredictDataFrom(filename string) (matrix.Matrix, error)
}

// DatasetReaders contains a map of the reader types which have strings as keys.
// The API works as following: the string in the json config file creates an instance
// of a given reader. Any reader can be created by satisfying the Reader trait and appending
// the key, value to the map.
var DatasetReaders = map[string]Reader{
	"mnist": Mnist{},
}
