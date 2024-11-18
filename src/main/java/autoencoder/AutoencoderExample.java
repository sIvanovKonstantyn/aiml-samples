package autoencoder;

import java.util.Arrays;

// Utility functions
class Utils {
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double sig = sigmoid(x);
        return sig * (1 - sig);
    }
}

// Layer class for simplicity
class Layer {
    double[][] weights;
    double[] biases;

    public Layer(int inputSize, int outputSize) {
        weights = new double[inputSize][outputSize];
        biases = new double[outputSize];

        // Random weight and bias initialization
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = Math.random() - 0.5;
            }
        }
        for (int j = 0; j < outputSize; j++) {
            biases[j] = Math.random() - 0.5;
        }
    }

    public double[] forward(double[] inputs) {
        double[] outputs = new double[biases.length];
        for (int j = 0; j < biases.length; j++) {
            outputs[j] = biases[j];
            for (int i = 0; i < inputs.length; i++) {
                outputs[j] += inputs[i] * weights[i][j];
            }
            outputs[j] = Utils.sigmoid(outputs[j]);
        }
        return outputs;
    }
}

// Autoencoder
class Autoencoder {
    private Layer encoder;
    private Layer decoder;

    public Autoencoder(int inputSize, int bottleneckSize) {
        encoder = new Layer(inputSize, bottleneckSize);
        decoder = new Layer(bottleneckSize, inputSize);
    }

    public double[] encode(double[] input) {
        return encoder.forward(input);
    }

    public double[] decode(double[] bottleneck) {
        return decoder.forward(bottleneck);
    }

    public double[] forward(double[] input) {
        double[] bottleneck = encode(input);
        return decode(bottleneck);
    }
}

public class AutoencoderExample {
    public static void main(String[] args) {
        // Example input data: 4-dimensional vector
        double[] input = {0.2, 0.8, 0.4, 0.6};

        // Initialize Autoencoder: input size = 4, bottleneck size = 2
        int inputSize = 4;
        int bottleneckSize = 2;
        Autoencoder autoencoder = new Autoencoder(inputSize, bottleneckSize);

        // Forward pass
        double[] reconstructed = autoencoder.forward(input);

        // Print original and reconstructed data
        System.out.println("Original Input: " + Arrays.toString(input));
        System.out.println("Reconstructed Output: " + Arrays.toString(reconstructed));
    }
}
