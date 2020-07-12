# fcnn

[![GoDoc](https://godoc.org/github.com/jmpargana/fcnn?status.svg)](https://godoc.org/github.com/jmpargana/fcnn)
[![Build Status](https://travis-ci.org/jmpargana/ged.svg?branch=master)](https://travis-ci.org/jmpargana/fcnn)
[![Status](https://travis-ci.org/jmpargana/matrix.svg?branch=master)](https://travis-ci.org/jmpargana/matrix)
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](LICENSE)

fcnn is a scalable fully connected neural network completely written from scratch
              using the Go Standard Library and an third party library to display the progress bar.
            fcnn takes advantage of Go's concurrency patterns to parallelize matrix multiplication
            and all operations a neural network can perform non sequentially. fcnn was also
            developed following some TDD principles. The GIF shows the binary training a
            config file with the MNIST dataset, but any other model can be trained as well, by
            defining a different dataset reader that satisfies the Reader interface.

![](https://s3.eu-central-1.amazonaws.com/jmpargana.github.io/fcnn.gif)

## Usage


Install the binary:

```sh
go get github.com/jmpargana/fcnn
```

You can enhance it with your own dataset readers, they just need to implement two functions *DataFrom(images, labels string) ([]Instance, error)* and *PredictDataFrom(filename string) (matrix.Matrix, error)* as described in the *readers/reader_type.go* file. 


## License


[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT License](LICENSE)**
- Copyright 2020 © João Pargana
