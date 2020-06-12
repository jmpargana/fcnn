package layer

// InSize returns expected input size vector.
// It is needed to check for valid input in the ForwProp method.
func (l *Layer) InSize() int {
	return l.Weights.NumCols
}

// OutSize returns the expected output size vector.
// Both these methods need to be public so they are accessible from
// the multilayerperceptron module.
func (l *Layer) OutSize() int {
	return l.Weights.NumRows
}
