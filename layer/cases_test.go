package layer

import "github.com/jmpargana/matrix"

type layerTestStruct struct {
	actFn   string
	inSize  int
	outSize int
}

type forwPropsTestStruct struct {
	actFn                 string
	in, out, sum, weights matrix.Matrix
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
