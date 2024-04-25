package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"
)


func main() {
	// 748 inputs 28 x 28 pixels, each pixel is an input
	// 100 hidden nodes
	// 10 outputs - digits 0 to 9
	// 0.1 is the learnign rate

	net := CreateNetwork(784, 200, 10, 0.1)

	mnist:= flat.String("mnist", "", "Either train or predict to evaluate neural network")
	flag.Parse()

	switch *mnist {
	case "train":
		mnistTrain(&net)
		save(net)
	case "predict":
		load(&net)
		mnistPredict(&net)
	default:
		// pass
	}


}

func mnistTrain(net *Network) {
	rand.Seed(time.Now().UTC.UnixNano())
	t1 := time.Now()

	for epochs := 0; epochs < 5; epochs++ {
		testFile, _ := os.Open("mnist_dataset/mnist_train.csv")
		r := csv.newReader(bufio.NewReader(testFile))

		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}

			inputs := make([]float64, net.inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = (x / 255.0 * 0.99) + 0.01
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.01
			}

			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.99

			net.Train(inputs, targets)
		}
		testFile.Close()
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train %s\n", elapsed)
}

func mnistPredict(net *Network) {
	t1 := time.Now()
	checkFile, _ := os.Open("mnist_dataset/mnist_test.csv")
	defer checkFile.Close()

	score := 0
	r := csv.NewReader(bufio.NewReader(checkFile))

	for {
		
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([float64, net.inputs])
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ :=str.conv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.99) + 0.01
		}

		outputs := net.Predict(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < net.outputs; i++ {
			if outputs.At(i, 0) > highest{
				best = i
				highest = outputs.At(i, 0)

			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
	}
	elapsed :=time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
	fmt.Println("score:", score)
}

// print out images on ITerm2; equivalent to imgcat on iTerm2????
func printImage(img image.Image) {
	var buf bytes.Buffer
	png.Encode(&buf, img)
	imgBase64Str := base64.StdEncoding.EncodeToString(buf.Bytes())
	fmt.Printf("\x1b]1337;File=inline=1:%s\a\n", imgBase64Str)
}


// get the file as an image
func getImage(filePath string) image.Image {
	imgFile, err :=os.Open(filePath)
	defer imgFile.close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, _, err := image.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}
	return img
}