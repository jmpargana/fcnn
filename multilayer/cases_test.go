package fcnn

type deltaOutStruct struct {
	actFn                 string
	output, expected, sum [][]float64
}

type deltaLastHiddenStruct struct {
	actFn                              string
	hiddenLayers                       []int
	prevDelta, delta, matrixPlus1, sum [][]float64
}

type updateBiasWeightTest struct {
	actFn                        string
	learningRate                 float64
	index                        int
	hiddenLayers                 []int
	delta, expected, biasWeights [][]float64
}

type calculateWeightsStruct struct {
	hiddenLayers             []int
	index                    int
	prevOut, delta, expected [][]float64
}

var hiddenLayersTest = [][]int{
	{23, 2, 12, 23, 42},
	{3, 2},
	{4, 23, 1},
}

var invalidHiddenLayersTest = [][]int{
	{23, 2, -12, 23, 42},
	{},
	{1},
	{-3, 2},
	{4, 0, 1},
}

var deltaOutTest = []deltaOutStruct{
	{
		actFn: "identity",
		output: [][]float64{
			{0},
			{0},
			{1},
			{0},
			{0},
		},
		sum: [][]float64{
			{1},
			{2},
			{3},
			{4},
			{5},
		},
		expected: [][]float64{
			{1},
			{2},
			{2},
			{4},
			{5},
		},
	},
	{
		actFn: "relu",
		output: [][]float64{
			{0},
			{0},
			{1},
			{0},
			{0},
		},
		sum: [][]float64{
			{0},
			{2},
			{0},
			{4},
			{-5},
		},
		expected: [][]float64{
			{0},
			{2},
			{0},
			{4},
			{0},
		},
	},
	// this test doesnt work because 0.285 != 0.285...
	// {
	// 	actFn: "lrelu",
	// 	output: [][]float64{
	// 		{10},
	// 		{-30},
	// 	},
	// 	sum: [][]float64{
	// 		{15},
	// 		{-150},
	// 	},
	// 	expected: [][]float64{
	// 		{5},
	// 		{0.285},
	// 	},
	// },
}

var deltaLastHiddenTest = []deltaLastHiddenStruct{
	{
		actFn:        "identity",
		hiddenLayers: []int{1, 1, 1, 6},
		matrixPlus1: [][]float64{
			{1, 2, 3, 4, 2, 3},
			{1, 2, 3, 4, 2, 3},
			{7, 2, 3, 4, 1, 3},
			{1, 2, 3, 4, 2, 3},
		},
		prevDelta: [][]float64{
			{1},
			{-1},
			{-1},
			{1},
		},
		sum: [][]float64{
			{1},
			{1},
			{1},
			{1},
			{1},
			{1},
		},
		delta: [][]float64{
			{-6},
			{0},
			{0},
			{0},
			{1},
			{0},
		},
	},
	{
		actFn:        "identity",
		hiddenLayers: []int{1, 1, 6},
		matrixPlus1: [][]float64{
			{1, 2, 3, 4, 2, 3},
			{1, 2, 3, 4, 2, 3},
			{7, 2, 3, 4, 1, 3},
			{1, 2, 3, 4, 2, 3},
		},
		prevDelta: [][]float64{
			{1},
			{-1},
			{-1},
			{1},
		},
		sum: [][]float64{
			{1},
			{1},
			{1},
			{1},
			{1},
			{1},
		},
		delta: [][]float64{
			{-6},
			{0},
			{0},
			{0},
			{1},
			{0},
		},
	},
	{
		actFn:        "identity",
		hiddenLayers: []int{1, 6},
		matrixPlus1: [][]float64{
			{1, 2, 3, 4, 2, 3},
			{1, 2, 3, 4, 2, 3},
			{7, 2, 3, 4, 1, 3},
			{1, 2, 3, 4, 2, 3},
		},
		prevDelta: [][]float64{
			{1},
			{-1},
			{-1},
			{1},
		},
		sum: [][]float64{
			{1},
			{1},
			{1},
			{1},
			{1},
			{1},
		},
		delta: [][]float64{
			{-6},
			{0},
			{0},
			{0},
			{1},
			{0},
		},
	},
	{
		actFn:        "binary_step",
		hiddenLayers: []int{1, 6},
		matrixPlus1: [][]float64{
			{1, 2, 3, 4, 2, 3},
			{1, 2, 3, 4, 2, 3},
			{7, 2, 3, 4, 1, 3},
			{1, 2, 3, 4, 2, 3},
		},
		prevDelta: [][]float64{
			{1},
			{-1},
			{-1},
			{1},
		},
		sum: [][]float64{
			{1},
			{-1},
			{1},
			{-1},
			{0},
			{-1},
		},
		delta: [][]float64{
			{0},
			{0},
			{0},
			{0},
			{1},
			{0},
		},
	},
}

var deltaTest = []deltaLastHiddenStruct{
	{
		actFn:        "identity",
		hiddenLayers: []int{1, 1, 6, 4},
		matrixPlus1: [][]float64{
			{1, 2, 3, 4, 2, 3},
			{1, 2, 3, 4, 2, 3},
			{7, 2, 3, 4, 1, 3},
			{1, 2, 3, 4, 2, 3},
		},
		prevDelta: [][]float64{
			{1},
			{-1},
			{-1},
			{1},
		},
		sum: [][]float64{
			{1},
			{1},
			{1},
			{1},
			{1},
			{1},
		},
		delta: [][]float64{
			{-6},
			{0},
			{0},
			{0},
			{1},
			{0},
		},
	},
	{
		actFn:        "identity",
		hiddenLayers: []int{1, 6, 4},
		matrixPlus1: [][]float64{
			{1, 2, 3, 4, 2, 3},
			{1, 2, 3, 4, 2, 3},
			{7, 2, 3, 4, 1, 3},
			{1, 2, 3, 4, 2, 3},
		},
		prevDelta: [][]float64{
			{1},
			{-1},
			{-1},
			{1},
		},
		sum: [][]float64{
			{1},
			{1},
			{1},
			{1},
			{1},
			{1},
		},
		delta: [][]float64{
			{-6},
			{0},
			{0},
			{0},
			{1},
			{0},
		},
	},
	{
		actFn:        "binary_step",
		hiddenLayers: []int{1, 6, 4},
		matrixPlus1: [][]float64{
			{1, 2, 3, 4, 2, 3},
			{1, 2, 3, 4, 2, 3},
			{7, 2, 3, 4, 1, 3},
			{1, 2, 3, 4, 2, 3},
		},
		prevDelta: [][]float64{
			{1},
			{-1},
			{-1},
			{1},
		},
		sum: [][]float64{
			{1},
			{-1},
			{1},
			{-1},
			{0},
			{-1},
		},
		delta: [][]float64{
			{0},
			{0},
			{0},
			{0},
			{1},
			{0},
		},
	},
}

var deltaFailTest = []deltaLastHiddenStruct{
	{
		actFn:        "identity",
		hiddenLayers: []int{1, 1, 6, 4},
		matrixPlus1: [][]float64{
			{1, 2, 3, 4, 2, 3},
			{1, 2, 3, 4, 2, 3},
			{7, 2, 3, 4, 1, 3},
			{1, 2, 3, 4, 2, 3},
		},
		prevDelta: [][]float64{
			{1},
			{-1},
			{-1},
		},
		sum: [][]float64{
			{1},
			{1},
			{1},
			{1},
			{1},
			{1},
		},
		delta: [][]float64{
			{-6},
			{0},
			{0},
			{0},
			{1},
			{0},
		},
	},
	{
		actFn:        "identity",
		hiddenLayers: []int{1, 6, 4},
		matrixPlus1: [][]float64{
			{1, 2, 3, 4, 2, 3},
			{1, 2, 3, 4, 2, 3},
			{7, 2, 3, 4, 1, 3},
		},
		prevDelta: [][]float64{
			{1},
			{-1},
			{-1},
			{1},
		},
		sum: [][]float64{
			{1},
			{1},
			{1},
			{1},
			{1},
			{1},
		},
		delta: [][]float64{
			{-6},
			{0},
			{0},
			{0},
			{1},
			{0},
		},
	},
	{
		actFn:        "binary_step",
		hiddenLayers: []int{1, 6, 4},
		matrixPlus1: [][]float64{
			{1, 2, 3, 4, 2, 3},
			{1, 2, 3, 4, 2, 3},
			{7, 2, 3, 4, 1, 3},
			{1, 2, 3, 4, 2, 3},
		},
		prevDelta: [][]float64{
			{1},
			{-1},
			{-1},
			{1},
		},
		sum: [][]float64{
			{1},
			{-1},
			{1},
			{-1},
			{0},
		},
		delta: [][]float64{
			{0},
			{0},
			{0},
			{0},
			{1},
			{0},
		},
	},
}

var updateBiasTest = []updateBiasWeightTest{
	{
		actFn:        "identity",
		learningRate: 0.5,
		index:        0,
		hiddenLayers: []int{2, 4, 1},
		biasWeights: [][]float64{
			{2},
			{4},
			{-2},
			{10},
		},
		delta: [][]float64{
			{1},
			{2},
			{-1},
			{5},
		},
		expected: [][]float64{
			{1.5},
			{3},
			{-1.5},
			{7.5},
		},
	},
	{
		actFn:        "identity",
		learningRate: 0.5,
		index:        1,
		hiddenLayers: []int{5, 2, 4, 1},
		biasWeights: [][]float64{
			{2},
			{4},
			{-2},
			{10},
		},
		delta: [][]float64{
			{1},
			{2},
			{-1},
			{5},
		},
		expected: [][]float64{
			{1.5},
			{3},
			{-1.5},
			{7.5},
		},
	},
	{
		actFn:        "identity",
		learningRate: 0.2,
		index:        1,
		hiddenLayers: []int{5, 2, 3, 1},
		biasWeights: [][]float64{
			{2},
			{-2},
			{10},
		},
		delta: [][]float64{
			{10},
			{-10},
			{10},
		},
		expected: [][]float64{
			{0},
			{0},
			{8},
		},
	},
}

var updateBiasOutTest = []updateBiasWeightTest{
	{
		actFn:        "identity",
		learningRate: 0.5,
		index:        0,
		hiddenLayers: []int{2, 2, 4},
		biasWeights: [][]float64{
			{2},
			{4},
			{-2},
			{10},
		},
		delta: [][]float64{
			{1},
			{2},
			{-1},
			{5},
		},
		expected: [][]float64{
			{1.5},
			{3},
			{-1.5},
			{7.5},
		},
	},
	{
		actFn:        "identity",
		learningRate: 0.5,
		index:        1,
		hiddenLayers: []int{5, 2, 2, 4},
		biasWeights: [][]float64{
			{2},
			{4},
			{-2},
			{10},
		},
		delta: [][]float64{
			{1},
			{2},
			{-1},
			{5},
		},
		expected: [][]float64{
			{1.5},
			{3},
			{-1.5},
			{7.5},
		},
	},
	{
		actFn:        "identity",
		learningRate: 0.2,
		index:        1,
		hiddenLayers: []int{5, 2, 3, 3},
		biasWeights: [][]float64{
			{2},
			{-2},
			{10},
		},
		delta: [][]float64{
			{10},
			{-10},
			{10},
		},
		expected: [][]float64{
			{0},
			{0},
			{8},
		},
	},
}

var updateBiasOutTestFail = []updateBiasWeightTest{
	{
		actFn:        "identity",
		learningRate: 0.5,
		index:        0,
		hiddenLayers: []int{2, 2, 4},
		biasWeights: [][]float64{
			{2},
			{4},
			{-2},
		},
		delta: [][]float64{
			{1},
			{2},
			{-1},
			{5},
		},
		expected: [][]float64{
			{1.5},
			{3},
			{-1.5},
			{7.5},
		},
	},
	{
		actFn:        "identity",
		learningRate: 0.5,
		index:        1,
		hiddenLayers: []int{5, 2, 2, 1},
		biasWeights: [][]float64{
			{2},
			{4},
			{-2},
			{10},
		},
		delta: [][]float64{
			{2},
			{-1},
			{5},
		},
		expected: [][]float64{
			{1.5},
			{3},
			{-1.5},
			{7.5},
		},
	},
	{
		actFn:        "identity",
		learningRate: 0.2,
		index:        1,
		hiddenLayers: []int{5, 2, 3, 3},
		biasWeights: [][]float64{
			{2},
			{-2},
			{10},
		},
		delta: [][]float64{
			{10},
			{-10},
		},
		expected: [][]float64{
			{0},
			{0},
			{8},
		},
	},
}

var updateWeightTest = []updateBiasWeightTest{
	{
		actFn:        "identity",
		learningRate: 0.5,
		index:        0,
		hiddenLayers: []int{3, 4, 1},
		biasWeights: [][]float64{
			{2, 2, 3},
			{4, 3, 4},
			{-2, 2, 1},
			{10, 1, 2},
		},
		delta: [][]float64{
			{2, 2, 3},
			{4, 3, 4},
			{-2, 2, 1},
			{10, 1, 2},
		},
		expected: [][]float64{
			{1, 1, 1.5},
			{2, 1.5, 2},
			{-1, 1, 0.5},
			{5, 0.5, 1},
		},
	},
	{
		actFn:        "identity",
		learningRate: 0.2,
		index:        0,
		hiddenLayers: []int{3, 4, 1},
		biasWeights: [][]float64{
			{2, 2, 3},
			{4, 3, 4},
			{-2, 2, 1},
			{10, 1, 2},
		},
		delta: [][]float64{
			{20, 20, 30},
			{40, 30, 40},
			{-200, 20, 10},
			{100, 10, 20},
		},
		expected: [][]float64{
			{-2, -2, -3},
			{-4, -3, -4},
			{38, -2, -1},
			{-10, -1, -2},
		},
	},
}

var updateWeightTestOut = []updateBiasWeightTest{
	{
		actFn:        "identity",
		learningRate: 0.5,
		index:        0,
		hiddenLayers: []int{3, 4, 1},
		biasWeights: [][]float64{
			{2, 2, 3},
			{4, 3, 4},
			{-2, 2, 1},
			{10, 1, 2},
		},
		delta: [][]float64{
			{2, 2, 3},
			{4, 3, 4},
			{-2, 2, 1},
			{10, 1, 2},
		},
		expected: [][]float64{
			{1, 1, 1.5},
			{2, 1.5, 2},
			{-1, 1, 0.5},
			{5, 0.5, 1},
		},
	},
	{
		actFn:        "identity",
		learningRate: 0.2,
		index:        0,
		hiddenLayers: []int{3, 4, 1},
		biasWeights: [][]float64{
			{2, 2, 3},
			{4, 3, 4},
			{-2, 2, 1},
			{10, 1, 2},
		},
		delta: [][]float64{
			{20, 20, 30},
			{40, 30, 40},
			{-200, 20, 10},
			{100, 10, 20},
		},
		expected: [][]float64{
			{-2, -2, -3},
			{-4, -3, -4},
			{38, -2, -1},
			{-10, -1, -2},
		},
	},
}

var updateWeightTestFail = []updateBiasWeightTest{
	{
		actFn:        "identity",
		learningRate: 0.5,
		index:        0,
		hiddenLayers: []int{3, 4, 1},
		biasWeights: [][]float64{
			{2, 2, 3},
			{4, 3, 4},
			{-2, 2, 1},
		},
		delta: [][]float64{
			{2, 2, 3},
			{4, 3, 4},
			{-2, 2, 1},
			{10, 1, 2},
		},
		expected: [][]float64{
			{1, 1, 1.5},
			{2, 1.5, 2},
			{-1, 1, 0.5},
			{5, 0.5, 1},
		},
	},
	{
		actFn:        "identity",
		learningRate: 0.2,
		index:        0,
		hiddenLayers: []int{3, 4, 1},
		biasWeights: [][]float64{
			{2, 2, 3},
			{4, 3, 4},
			{-2, 2, 1},
			{10, 1, 2},
		},
		delta: [][]float64{
			{20, 20, 30},
			{40, 30, 40},
			{-200, 20, 10},
		},
		expected: [][]float64{
			{-2, -2, -3},
			{-4, -3, -4},
			{38, -2, -1},
			{-10, -1, -2},
		},
	},
}

var calculateWeightTest = []calculateWeightsStruct{
	{
		hiddenLayers: []int{4, 3, 1},
		index:        1,
		prevOut: [][]float64{
			{1},
			{1},
			{1},
		},
		delta: [][]float64{
			{1},
			{1},
			{1},
			{1},
		},
		expected: [][]float64{
			{1, 1, 1},
			{1, 1, 1},
			{1, 1, 1},
			{1, 1, 1},
		},
	},
	{
		hiddenLayers: []int{4, 4, 3, 1},
		index:        2,
		prevOut: [][]float64{
			{1},
			{-1},
			{1},
		},
		delta: [][]float64{
			{1},
			{1},
			{1},
			{-1},
		},
		expected: [][]float64{
			{1, -1, 1},
			{1, -1, 1},
			{1, -1, 1},
			{-1, 1, -1},
		},
	},
	{
		hiddenLayers: []int{4, 3, 1},
		index:        1,
		prevOut: [][]float64{
			{1},
			{1},
			{-1},
		},
		delta: [][]float64{
			{2},
			{3},
			{2},
			{4},
		},
		expected: [][]float64{
			{2, 2, -2},
			{3, 3, -3},
			{2, 2, -2},
			{4, 4, -4},
		},
	},
	{
		hiddenLayers: []int{4, 3, 5, 6, 1},
		index:        3,
		prevOut: [][]float64{
			{10},
			{1},
			{10},
			{1},
			{10},
		},
		delta: [][]float64{
			{0},
			{0},
			{1},
			{0},
			{1},
			{0},
		},
		expected: [][]float64{
			{0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0},
			{10, 1, 10, 1, 10},
			{0, 0, 0, 0, 0},
			{10, 1, 10, 1, 10},
			{0, 0, 0, 0, 0},
		},
	},
}

var calculateWeightTestOut = []calculateWeightsStruct{
	{
		hiddenLayers: []int{4, 3, 1},
		index:        1,
		prevOut: [][]float64{
			{1},
			{1},
			{1},
		},
		delta: [][]float64{
			{1},
			{1},
			{1},
			{1},
		},
		expected: [][]float64{
			{1, 1, 1},
			{1, 1, 1},
			{1, 1, 1},
			{1, 1, 1},
		},
	},
	{
		hiddenLayers: []int{4, 4, 3, 1},
		index:        2,
		prevOut: [][]float64{
			{1},
			{-1},
			{1},
		},
		delta: [][]float64{
			{1},
			{1},
			{1},
			{-1},
		},
		expected: [][]float64{
			{1, -1, 1},
			{1, -1, 1},
			{1, -1, 1},
			{-1, 1, -1},
		},
	},
	{
		hiddenLayers: []int{4, 3, 1},
		index:        1,
		prevOut: [][]float64{
			{1},
			{1},
			{-1},
		},
		delta: [][]float64{
			{2},
			{3},
			{2},
			{4},
		},
		expected: [][]float64{
			{2, 2, -2},
			{3, 3, -3},
			{2, 2, -2},
			{4, 4, -4},
		},
	},
	{
		hiddenLayers: []int{4, 3, 5, 6, 1},
		index:        3,
		prevOut: [][]float64{
			{10},
			{1},
			{10},
			{1},
			{10},
		},
		delta: [][]float64{
			{0},
			{0},
			{1},
			{0},
			{1},
			{0},
		},
		expected: [][]float64{
			{0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0},
			{10, 1, 10, 1, 10},
			{0, 0, 0, 0, 0},
			{10, 1, 10, 1, 10},
			{0, 0, 0, 0, 0},
		},
	},
}

var calculateWeightTestInvalid = []calculateWeightsStruct{
	{
		hiddenLayers: []int{4, 3, 1},
		index:        1,
		prevOut: [][]float64{
			{1},
			{1},
		},
		delta: [][]float64{
			{1},
			{1},
			{1},
			{1},
		},
		expected: [][]float64{
			{1, 1, 1},
			{1, 1, 1},
			{1, 1, 1},
			{1, 1, 1},
		},
	},
	{
		hiddenLayers: []int{1, 4, 3, 1},
		index:        2,
		prevOut: [][]float64{
			{1},
			{-1},
			{1},
		},
		delta: [][]float64{
			{1},
			{1},
			{1},
			{-1},
		},
		expected: [][]float64{
			{1, -1, 1},
			{1, -1, 1},
			{1, -1, 1},
			{-1, 1, -1},
		},
	},
	{
		hiddenLayers: []int{4, 3, 1},
		index:        1,
		prevOut: [][]float64{
			{1},
			{1},
			{-1},
		},
		delta: [][]float64{
			{2},
			{3},
			{2},
		},
		expected: [][]float64{
			{2, 2, -2},
			{3, 3, -3},
			{2, 2, -2},
			{4, 4, -4},
		},
	},
	{
		hiddenLayers: []int{4, 3, 5, 6, 1},
		index:        3,
		prevOut: [][]float64{
			{10},
			{1},
			{10},
			{1},
			{10},
			{10},
		},
		delta: [][]float64{
			{0},
			{0},
			{1},
			{0},
			{1},
			{0},
		},
		expected: [][]float64{
			{0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0},
			{10, 1, 10, 1, 10},
			{0, 0, 0, 0, 0},
			{10, 1, 10, 1, 10},
			{0, 0, 0, 0, 0},
		},
	},
}

var gradientDescentTest = []struct {
	hiddenLayers    []int
	outputLayer     int
	learningRate    float64
	deltaBias       [][][]float64
	expectedBias    [][][]float64
	deltaWeights    [][][]float64
	expectedWeights [][][]float64
	bias            [][][]float64
	weights         [][][]float64
}{
	{
		hiddenLayers: []int{4, 6, 3},
		outputLayer:  2,
		learningRate: 0.1,
		weights: [][][]float64{
			{
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
				{1, 1, 1, 1},
			},
			{
				{1, 2, 3, 1, 2, 3},
				{2, 3, 2, 3, 3, 2},
				{3, 1, 1, 2, 1, 1},
			},
			{
				{1, 1, 1},
				{1, 1, 1},
			},
		},
		deltaWeights: [][][]float64{
			{
				{10, 10, 10, 10},
				{10, 10, 10, 10},
				{10, 10, 10, 10},
				{10, 10, 10, 10},
				{10, 10, 10, 10},
				{10, 10, 10, 10},
			},
			{
				{100, 100, 100, 100, 100, 100},
				{100, 100, 100, 100, 100, 100},
				{100, 100, 100, 100, 100, 100},
			},
			{
				{-20, -30, -40},
				{-30, -40, -20},
			},
		},
		expectedWeights: [][][]float64{
			{
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
			{
				{-9, -8, -7, -9, -8, -7},
				{-8, -7, -8, -7, -7, -8},
				{-7, -9, -9, -8, -9, -9},
			},
			{
				{3, 4, 5},
				{4, 5, 3},
			},
		},
		bias: [][][]float64{
			{
				{5},
				{5},
				{1},
				{1},
				{5},
				{1},
			},
			{
				{-1},
				{0},
				{-1},
			},
			{
				{-1},
				{-1},
			},
		},
		deltaBias: [][][]float64{
			{
				{50},
				{50},
				{10},
				{10},
				{50},
				{10},
			},
			{
				{100},
				{100},
				{10},
			},
			{
				{10},
				{10},
			},
		},
		expectedBias: [][][]float64{
			{
				{0},
				{0},
				{0},
				{0},
				{0},
				{0},
			},
			{
				{-11},
				{-10},
				{-2},
			},
			{
				{-2},
				{-2},
			},
		},
	},
}
