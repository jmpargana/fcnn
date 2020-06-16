package readers

import (
	"bufio"
	"os"

	"github.com/jmpargana/matrix"
)

type Instance struct {
	Image, Label matrix.Matrix
}

func Mnist(train, test string) ([]Instance, error) {
	trainFile, err := os.Open(train)
	if err != nil {
		return nil, err
	}
	testFile, err := os.Open(test)
	if err != nil {
		return nil, err
	}

	trainScanner := bufio.NewScanner(trainFile)
	testScanner := bufio.NewScanner(testFile)

	// just space holder so package compiles
	_, _ = trainScanner, testScanner

	return []Instance{}, nil
}
