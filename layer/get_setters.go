package layer

// inSize returns expected input size vector.
// It is needed to check for valid input in the ForwProp method.
func (l *Layer) inSize() int {
	return l.weights.NumRows
}

func (l *Layer) outSize() int {
	return l.weights.NumCols
}
