package multilayer

import "testing"

func TestDecoder(t *testing.T) {

	tt := map[string]struct {
		hLayers         []int
		oLayer          int
		actFn, outActFn string
		batch, epoch    int
		lRate           float64
	}{
		"simple mlp": {
			[]int{3, 4, 6},
			10,
			"relu",
			"relu",
			1,
			1,
			0.2,
		},
	}

	for name, tc := range tt {
		t.Run(name, func(t *testing.T) {
			m, err := New(tc.hLayers, tc.oLayer, tc.actFn, tc.outActFn, tc.batch, tc.epoch, tc.lRate)
			if err != nil {
				t.Errorf("failed in constructor: %v", err)
			}

			mlp := MultiLayerPerceptron{}

			data, err := m.MarshalBinary()
			if err != nil {
				t.Errorf("failed encoding: %v", err)
			}

			if err := mlp.UnmarshalBinary(data); err != nil {
				t.Errorf("failed decoding: %v", err)
			}

			if !m.Equal(mlp) {
				t.Errorf("got:\n%vwanted:\n%v", mlp, m)
			}
		})
	}
}
