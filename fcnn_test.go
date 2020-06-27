package main

// this test doesn't work on travis
// func TestLoadSaveNN(t *testing.T) {
// 	modelName := "tmp"

// 	conf, err := parseConfig("examples/mnist_small_config.json")
// 	if err != nil {
// 		t.Errorf("failed parsing the example file: %v", err)
// 	}

// 	nn, err := fromParsedConfig(conf)
// 	if err != nil {
// 		t.Errorf("failed creating nn with provided configuration: %v", err)
// 	}

// 	if err := saveNetwork(nn, modelName, ""); err != nil {
// 		t.Errorf("failed saving nn: %v", err)
// 	}

// 	loadedNN, err := loadModel(modelName)
// 	if err != nil {
// 		t.Errorf("failed loading nn: %v", err)
// 	}

// 	if !loadedNN.Equal(nn) {
// 		t.Errorf("uploaded nn is different than saved\ngot:\n%v\nwant:\n%v\n", loadedNN, nn)
// 	}
// }
