package layer

import (
	"fmt"
	"testing"

	"github.com/jmpargana/matrix"
)

func TestDIdentity(t *testing.T) {
	for _, test := range dIdentityTest {
		in := matrix.NewFrom(test.in)
		out := matrix.NewFrom(test.out)
		got, err := DerivativeFunctions["identity"](in)
		if err != nil {
			t.Errorf("something failed: %v", err)
		}

		if !got.Equal(out) {
			t.Errorf("\ninput: %dx%d matrix:\n%v\ngot: %dx%d matrix:\n%v\nexpected: %dx%d matrix:\n%v\n", in.NumRows, in.NumCols, in, got.NumRows, got.NumCols, got, out.NumRows, out.NumCols, out)
		}
	}
}

func TestDRelu(t *testing.T) {
	for _, test := range dReluTest {
		in := matrix.NewFrom(test.in)
		out := matrix.NewFrom(test.out)
		got, err := DerivativeFunctions["relu"](in)
		if err != nil {
			t.Errorf("something failed: %v", err)
		}

		if !got.Equal(out) {
			for row := 0; row < got.NumRows; row++ {
				elemGot, _ := got.Get(row, 0)
				elemExpected, _ := out.Get(row, 0)
				fmt.Printf("got: %f, expected: %f\n", elemGot, elemExpected)
			}

			t.Errorf("\ninput: %dx%d matrix:\n%v\ngot: %dx%d matrix:\n%v\nexpected: %dx%d matrix:\n%v\n", in.NumRows, in.NumCols, in, got.NumRows, got.NumCols, got, out.NumRows, out.NumCols, out)
		}
	}

}

func TestDBinaryStep(t *testing.T) {
	for _, test := range dBinaryStepTest {
		in := matrix.NewFrom(test.in)
		out := matrix.NewFrom(test.out)
		got, err := DerivativeFunctions["binary_step"](in)
		if err != nil {
			t.Errorf("something failed: %v", err)
		}

		if !got.Equal(out) {
			t.Errorf("\ninput: %dx%d matrix:\n%v\ngot: %dx%d matrix:\n%v\nexpected: %dx%d matrix:\n%v\n", in.NumRows, in.NumCols, in, got.NumRows, got.NumCols, got, out.NumRows, out.NumCols, out)
		}
	}
}

func TestDLrelu(t *testing.T) {
	for _, test := range dLreluTest {
		in := matrix.NewFrom(test.in)
		out := matrix.NewFrom(test.out)
		got, err := DerivativeFunctions["lrelu"](in)
		if err != nil {
			t.Errorf("something failed: %v", err)
		}

		if !got.Equal(out) {
			for row := 0; row < got.NumRows; row++ {
				elemGot, _ := got.Get(row, 0)
				elemExpected, _ := out.Get(row, 0)
				fmt.Printf("got: %f, expected: %f\n", elemGot, elemExpected)
			}

			t.Errorf("\ninput: %dx%d matrix:\n%v\ngot: %dx%d matrix:\n%v\nexpected: %dx%d matrix:\n%v\n", in.NumRows, in.NumCols, in, got.NumRows, got.NumCols, got, out.NumRows, out.NumCols, out)
		}
	}

}

func TestDArcTan(t *testing.T) {
	for _, test := range dArcTanTest {
		in := matrix.NewFrom(test.in)
		out := matrix.NewFrom(test.out)
		got, err := DerivativeFunctions["arctan"](in)
		if err != nil {
			t.Errorf("something failed: %v", err)
		}

		if !got.Equal(out) {
			t.Errorf("\ninput: %dx%d matrix:\n%v\ngot: %dx%d matrix:\n%v\nexpected: %dx%d matrix:\n%v\n", in.NumRows, in.NumCols, in, got.NumRows, got.NumCols, got, out.NumRows, out.NumCols, out)
		}
	}

}
