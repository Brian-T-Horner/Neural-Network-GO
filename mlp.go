package main

//test change

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)


// 3 layer feedforward neural network (multi-layer perceptron)

type Network struct {
	inputs 			int
	hiddens 		int
	outputs 		int
	hiddenWeights 	*mat.Dense
	outputWeights 	*mat.Dense
	learningRate 	float64
}

// Creates a multi-layer perceptron
func CreateNetwork(input, hidden, output int, rate float64) (net Network) {
	net = Network {
		inputs:		input,
		hiddens: 	hidden,
		outputs:	output,
		learningRate: rate,
	}

	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randomArray(net.hiddens*net.outputs, float64(net.hiddens)))
	return

}

// Train the neural network
func (net *Network) Train(inputData []float64, targetData []float64) {
	//forward propogration
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	// find errors
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := subtract(targets, finalOutputs)
	hiddenErrors := dot(net.outputWeights.T(), outputErrors)

	// backpropagate
	net.outputWeights = add(net.outputWeights, 
		scale(net.learningRate,
			 dot(multiply(outputErrors, sigmoidPrime(finalOutputs)),
			 hiddenOutputs.T()))).(*mat.Dense)

	net.hiddenWeights = add(net.hiddenWeights,
	scale(net.learningRate,
		dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
		inputs.T()))).(*mat.Dense)
}

// Predict the values using the trained neural network
func (net Network) Predict(inputData []float64) mat.Matrix {
	//forward propagation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)
	return finalOutputs
}

// Gets data from the PNG file
func dataFromImage(filePath string) (pixels []float64) {
	//read the file
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	
	img, err := png.Decode(imgFile)
	
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}

	// create a greyscale image
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}

	// make a pixel array
	pixels = make([]float64, len(gray.Pix))
	// populating the pixel array and subtracting Pix from 255 because
	// MINST database was trained (in reverse?)
	//MINST is white on black so we need to subtract each pixel value from 255
	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.999) + 0.001
	}
	return
}


// takes in neural network and predicts the digit from an image file
// results are an array of probabilities where the index is the digit
func predictFromImage(net Network, path string) int {
	input := dataFromImage(path)
	output := net.Predict(input)
	matrixPrint(output)
	best := 0
	highest := 0.0
	for i := 0; i < net.outputs; i++ {
		if output.At(i, 0) > highest {
			best = i
			highest = output.At(i, 0)
		}
	}
	return best
}



// Sigmoid Prime function
func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1-m)
}

// Sigmoid function
func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}






// Helper functions 

// pretty print a Gonum matrix
func matrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}


// Save hidden and output weights
func save(net Network) {
	h, err := os.Create("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}
	o, err := os.Create("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
}

// Load saved hidden and output weights 
func load(net *Network) {
	h, err := os.Open("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
	return
}


// Create random array using a Uniform distribution of float64 nums
func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max : 1 /math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}

// Finds the size of matrix, creates it and preforms the operation
func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims() // Short variable declaration operator
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o

}

// apply a function to the matrix
func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

// multiple a matrix by a scalar (scale a matrix)
func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

// multiplies 2 functions together
func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

// add a function to/from another
func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

// subtract a function to/from another
func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

// add a scalar value to each element in matrix
func addScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i 
	}
	n := mat.NewDense(r, c, a)
	return add(m, n)
}