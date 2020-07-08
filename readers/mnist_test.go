package readers

import "testing"

func TestDataFrom(t *testing.T) {
	tt := map[string]struct {
		err, images, labels string
	}{
		"non existing filenames": {
			"invalid path",
			"",
			"",
		},
	}

	mnist := Mnist{}

	for name, tc := range tt {
		t.Run(name, func(t *testing.T) {
			got, err := mnist.DataFrom(tc.images, tc.labels)

			if tc.err != "" {
				if err == nil {
					t.Errorf("was supposed to fail here")
				}
			} else {
				if len(got) != 10000 {
					t.Errorf("wrong dataset size: %d", len(got))
				}
			}
		})
	}
}
