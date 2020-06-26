package layer

import (
	"bytes"
	"encoding/gob"
)

func (l *Layer) MarshalBinary() ([]byte, error) {
	w := wrapLayer{l.actFn, l.Weights, l.Output, l.Sum, l.Bias}

	buf := new(bytes.Buffer)
	if err := gob.NewEncoder(buf).Encode(&w); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (l *Layer) UnmarshalBinary(data []byte) error {
	w := wrapLayer{}

	reader := bytes.NewReader(data)
	if err := gob.NewDecoder(reader).Decode(&w); err != nil {
		return err
	}

	l.actFn = w.ActFn
	l.Weights = w.Weights
	l.Output = w.Output
	l.Sum = w.Sum
	l.Bias = w.Bias

	return nil
}
