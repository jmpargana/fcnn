package layer

type actFnTest struct {
	in  [][]float64
	out [][]float64
}

type layerTestStruct struct {
	actFn   string
	inSize  int
	outSize int
}

type forwPropsTestStruct struct {
	actFn                 string
	in, out, sum, weights [][]float64
}

type forwPropsWithBiasTestStruct struct {
	actFn                       string
	in, out, sum, weights, bias [][]float64
}

var forwPropsWithBiasTest = []forwPropsWithBiasTestStruct{
	{
		actFn: "identity",
		in: [][]float64{
			{2},
			{-3},
			{-1},
			{3},
		},
		out: [][]float64{
			{2},
			{11},
			{17},
		},
		sum: [][]float64{
			{2},
			{11},
			{17},
		},
		bias: [][]float64{
			{1},
			{4},
			{3},
		},
		weights: [][]float64{
			{2, 4, 3, 4},
			{3, 2, 2, 3},
			{3, 1, -2, 3},
		},
	},
	{
		actFn: "identity",
		in: [][]float64{
			{2},
			{-3},
			{-1},
			{3},
		},
		out: [][]float64{
			{0},
			{11},
			{11},
		},
		sum: [][]float64{
			{0},
			{11},
			{11},
		},
		bias: [][]float64{
			{-1},
			{4},
			{-3},
		},
		weights: [][]float64{
			{2, 4, 3, 4},
			{3, 2, 2, 3},
			{3, 1, -2, 3},
		},
	},
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

var dIdentityTest = []actFnTest{
	{
		[][]float64{
			{1},
			{-2},
			{-1},
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
			{1},
			{1},
			{1},
		},
	},
}

var dReluTest = []actFnTest{
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
var dBinaryStepTest = []actFnTest{
	{
		[][]float64{
			{1},
			{-2},
			{-1},
			{2},
		},
		[][]float64{
			{0},
			{0},
			{0},
			{0},
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
			{0},
			{0},
			{0},
			{0},
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
			{0},
		},
	},
}

var dLreluTest = []actFnTest{
	{
		[][]float64{
			{1},
			{-2},
			{-1},
			{2},
		},
		[][]float64{
			{0},
			{0.9996},
			{0.9999},
			{-3},
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
			{0},
			{-3},
			{-8},
			{-3},
		},
	},
	{
		[][]float64{
			{-1},
			{-2},
			{20},
		},
		[][]float64{
			{0.9999},
			{0.9996},
			{-399},
		},
	},
}
var dArcTanTest = []actFnTest{
	{
		[][]float64{
			{1},
			{-2},
			{-1},
			{2},
		},
		[][]float64{
			{0.5},
			{0.2},
			{0.5},
			{0.2},
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
			{0.50},
			{0.20},
			{0.10},
			{0.20},
		},
	},
	{
		[][]float64{
			{-1},
			{-2},
			{0},
		},
		[][]float64{
			{0.5},
			{0.2},
			{1},
		},
	},
}
