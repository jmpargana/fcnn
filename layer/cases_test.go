package layer

type layerTestStruct struct {
	actFn   string
	inSize  int
	outSize int
}

type forwPropsTestStruct struct {
	actFn                 string
	in, out, sum, weights [][]float64
}

var forwPropsTest = []forwPropsTestStruct{
	{
		actFn: "identity",
		in: [][]float64{
			{2},
			{-3},
			{-1},
			{3},
		},
		out: [][]float64{
			{1},
			{7},
			{14},
		},
		sum: [][]float64{
			{1},
			{7},
			{14},
		},
		weights: [][]float64{
			{2, 4, 3, 4},
			{3, 2, 2, 3},
			{3, 1, -2, 3},
		},
	},
	{
		actFn: "binary_step",
		in: [][]float64{
			{2},
			{-3},
			{-1},
			{6},
		},
		out: [][]float64{
			{1},
			{0},
			{1},
		},
		sum: [][]float64{
			{13},
			{-32},
			{11},
		},
		weights: [][]float64{
			{2, 4, 3, 4},
			{-3, 2, 2, -3},
			{-3, 1, -2, 3},
		},
	},
	{
		actFn: "relu",
		in: [][]float64{
			{2},
			{-3},
			{-1},
			{6},
		},
		out: [][]float64{
			{13},
			{0},
			{11},
		},
		sum: [][]float64{
			{13},
			{-32},
			{11},
		},
		weights: [][]float64{
			{2, 4, 3, 4},
			{-3, 2, 2, -3},
			{-3, 1, -2, 3},
		},
	},
	{
		actFn: "relu",
		in: [][]float64{
			{-52},
			{-3},
			{-1},
			{-233},
		},
		out: [][]float64{
			{0},
			{0},
			{0},
		},
		sum: [][]float64{
			{-1051},
			{-827},
			{-856},
		},
		weights: [][]float64{
			{2, 4, 3, 4},
			{3, -2, -22, 3},
			{3, 1, -2, 3},
		},
	},
	{
		actFn: "binary_step",
		in: [][]float64{
			{2},
			{-3},
			{-1},
			{3},
		},
		out: [][]float64{
			{1},
			{1},
			{1},
		},
		sum: [][]float64{
			{1},
			{21},
			{752},
		},
		weights: [][]float64{
			{2, 4, 3, 4},
			{3, 2, -12, 3},
			{333, -31, -2, -3},
		},
	},
}

var layersTest = []layerTestStruct{
	{
		"relu",
		1,
		1,
	},
	{
		"sigmoid",
		8,
		4,
	},
	{
		"softmax",
		8,
		8,
	},
	{
		"lrelu",
		10,
		100,
	},
	{
		"binary_step",
		40,
		23,
	},
	{
		"arctan",
		1,
		8,
	},
}

var layersInvalidTest = []layerTestStruct{
	{
		"unavailable",
		1,
		1,
	},
	{
		"sigmoid",
		0,
		4,
	},
	{
		"softmax",
		8,
		0,
	},
	{
		"lrelu",
		0,
		0,
	},
	{
		"binary_step",
		-1,
		23,
	},
	{
		"wrong",
		-1,
		-8,
	},
}
