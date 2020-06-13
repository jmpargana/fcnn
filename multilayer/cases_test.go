package fcnn

type deltaOutStruct struct {
	actFn                 string
	output, expected, sum [][]float64
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
