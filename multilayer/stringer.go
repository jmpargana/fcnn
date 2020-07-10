package multilayer

import "fmt"

func (m *MultiLayerPerceptron) String() (s string) {
	for i, hidden := range m.HiddenLayers {
		s += fmt.Sprintf("Hidden Layer nr: %d\n", i)
		s += fmt.Sprintln(hidden.String())
	}
	s += fmt.Sprintln("Output Layer!")
	s += fmt.Sprintln(m.outputLayer.String())
	return
}
