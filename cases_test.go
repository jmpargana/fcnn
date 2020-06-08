package fcnn

import . "github.com/jmpargana/matrix"

type actFnTest struct {
	in  [][]float64
	out [][]float64
}

type actFnTestMat struct {
	in  Matrix
	out Matrix
}

var identityTest = []actFnTest{
	{
		[][]float64{
			{1},
			{2},
			{1},
			{2},
		},
		[][]float64{
			{1},
			{2},
			{1},
			{2},
		},
	},
	{
		[][]float64{
			{1},
			{2},
			{3},
			{2},
		},
		[][]float64{
			{1},
			{2},
			{3},
			{2},
		},
	},
	{
		[][]float64{
			{1},
			{-2},
			{1},
		},
		[][]float64{
			{1},
			{-2},
			{1},
		},
	},
}

var binaryStepTest = []actFnTest{
	{
		[][]float64{
			{1},
			{-2},
			{-1},
			{2},
		},
		[][]float64{
			{1},
			{0},
			{0},
			{1},
		},
	},
	{
		[][]float64{
			{1},
			{2},
			{3},
			{2},
		},
		[][]float64{
			{1},
			{1},
			{1},
			{1},
		},
	},
	{
		[][]float64{
			{-1},
			{-2},
			{20},
		},
		[][]float64{
			{0},
			{0},
			{1},
		},
	},
}

var reluTest = []actFnTest{
	{
		[][]float64{
			{1},
			{-2},
			{-1},
			{2},
		},
		[][]float64{
			{1},
			{0},
			{0},
			{2},
		},
	},
	{
		[][]float64{
			{1},
			{2},
			{3},
			{2},
		},
		[][]float64{
			{1},
			{2},
			{3},
			{2},
		},
	},
	{
		[][]float64{
			{-1},
			{-2},
			{20},
		},
		[][]float64{
			{0},
			{0},
			{20},
		},
	},
}

var lReLUTest = []actFnTest{
	{
		[][]float64{
			{1},
			{-2},
			{-1},
			{2},
		},
		[][]float64{
			{1},
			{-0.02},
			{-0.01},
			{2},
		},
	},
	{
		[][]float64{
			{1},
			{2},
			{3},
			{2},
		},
		[][]float64{
			{1},
			{2},
			{3},
			{2},
		},
	},
	{
		[][]float64{
			{-1},
			{-2},
			{20},
		},
		[][]float64{
			{-0.01},
			{-0.02},
			{20},
		},
	},
}
