package fcnn

var hiddenLayersTest = [][]int{
	{23, 2, 12, 23, 42},
	{1},
	{3, 2},
	{4, 23, 1},
}

var invalidHiddenLayersTest = [][]int{
	{23, 2, -12, 23, 42},
	{},
	{-3, 2},
	{4, 0, 1},
}
