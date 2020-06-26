package layer

import "testing"

func TestMarshaling(t *testing.T) {
	tt := map[string]struct {
		actFn   string
		in, out int
	}{
		"smallest layer size":       {"relu", 1, 1},
		"vector weights":            {"relu", 1, 10},
		"vector weights transposed": {"relu", 10, 1},
		"regular layer":             {"relu", 4, 5},
	}

	for name, tc := range tt {
		t.Run(name, func(t *testing.T) {
			l, err := New(tc.actFn, tc.in, tc.out)
			got := Layer{}

			if err != nil {
				t.Errorf("failed in constructor: %v", err)
			}

			data, err := l.MarshalBinary()
			if err != nil {
				t.Errorf("shoudln't fail encoding: %v", err)
			}

			if err := got.UnmarshalBinary(data); err != nil {
				t.Errorf("shouldn't fail decoding: %v", err)
			}

			if !l.Equal(got) {
				t.Errorf("%s should be equal to %s", got, l)
			}
		})
	}
}
