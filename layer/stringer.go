package layer

import "fmt"

func (l *Layer) String() (s string) {
	s += fmt.Sprintf("Input: %d, Output: %d\nActivation Function: %s\n", l.InSize(), l.OutSize(), l.actFn)
	s += fmt.Sprintln("Weights")
	s += fmt.Sprint(l.Weights)
	s += fmt.Sprintln("Bias")
	s += fmt.Sprint(l.Bias)
	return
}
