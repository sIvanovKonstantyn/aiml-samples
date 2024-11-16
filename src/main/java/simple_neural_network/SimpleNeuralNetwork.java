package simple_neural_network;

import java.util.Random;
import java.util.Arrays;

// Class representing a single neuron
class Neuron {
    private double[] weights;
    private double bias;
    private double output; // Stores the neuron's output during forward pass

    public Neuron(int inputSize, Random random) {
        // Initialize weights and bias randomly
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = random.nextDouble() - 0.5;
        }
        bias = random.nextDouble() - 0.5;
    }

    // Forward pass for the neuron
    public double forward(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        output = relu(sum); // Apply activation function
        return output;
    }

    // ReLU activation function
    private double relu(double x) {
        return Math.max(0, x);
    }

    // Backward pass for gradient update
    public void backward(double[] inputs, double gradient, double learningRate) {
        double reluGradient = (output > 0) ? 1 : 0; // Derivative of ReLU
        double delta = gradient * reluGradient;

        // Update weights and bias
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * delta * inputs[i];
        }
        bias -= learningRate * delta;
    }

    public double[] getWeights() {
        return weights;
    }
}

// Class representing a layer of neurons
class Layer {
    private Neuron[] neurons;

    public Layer(int neuronCount, int inputSize, Random random) {
        neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputSize, random);
        }
    }

    // Forward pass for the layer
    public double[] forward(double[] inputs) {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].forward(inputs);
        }
        return outputs;
    }

    // Backward pass for the layer
    public void backward(double[] inputs, double[] gradients, double learningRate) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].backward(inputs, gradients[i], learningRate);
        }
    }

    public Neuron[] getNeurons() {
        return neurons;
    }
}

// Class representing the entire neural network
class SimpleNeuralNetwork {
    private Layer hiddenLayer;
    private Layer outputLayer;
    private double learningRate;

    public SimpleNeuralNetwork(int inputSize, int hiddenSize, double learningRate) {
        Random random = new Random();
        this.learningRate = learningRate;
        hiddenLayer = new Layer(hiddenSize, inputSize, random);
        outputLayer = new Layer(1, hiddenSize, random); // Single output neuron
    }

    // Forward pass through the entire network
    public double forward(double[] inputs) {
        double[] hiddenOutputs = hiddenLayer.forward(inputs);
        double[] output = outputLayer.forward(hiddenOutputs);
        return output[0];
    }

    // Train the network on a single data point
    public void train(double[] inputs, double target) {
        // Forward pass
        double[] hiddenOutputs = hiddenLayer.forward(inputs);
        double predicted = outputLayer.forward(hiddenOutputs)[0];

        // Calculate error and gradient for the output layer
        double outputError = predicted - target;
        double[] outputGradients = { outputError };

        // Backward pass
        outputLayer.backward(hiddenOutputs, outputGradients, learningRate);

        // Propagate error to hidden layer
        double[] hiddenGradients = new double[hiddenLayer.getNeurons().length];
        Neuron[] outputNeurons = outputLayer.getNeurons();
        for (int i = 0; i < hiddenGradients.length; i++) {
            hiddenGradients[i] = outputError * outputNeurons[0].getWeights()[i];
        }
        hiddenLayer.backward(inputs, hiddenGradients, learningRate);
    }

    public static void main(String[] args) {
        // XOR dataset
        double[][] inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        double[] targets = { 0, 1, 1, 0 };

        // Create and train the network
        SimpleNeuralNetwork nn = new SimpleNeuralNetwork(2, 2, 0.1);
        for (int epoch = 0; epoch < 10000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                nn.train(inputs[i], targets[i]);
            }
        }

        // Test the network
        System.out.println("Predictions:");
        for (int i = 0; i < inputs.length; i++) {
            double prediction = nn.forward(inputs[i]);
            System.out.printf("Input: %s, Prediction: %.4f%n", Arrays.toString(inputs[i]), prediction);
        }
    }
}

