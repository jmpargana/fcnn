package multilayer

import "fmt"

func (m *MultiLayerPerceptron) String() (s string) {
	for _, hidden := range m.HiddenLayers {
		s += fmt.Sprintln(hidden)
	}
	s += fmt.Sprintln(m.outputLayer)

	return
}
