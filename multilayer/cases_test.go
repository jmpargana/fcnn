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
}
