package cnn;

import java.util.Arrays;

// Utility class for common operations
class Utils {
    // ReLU activation function
    public static double relu(double x) {
        return Math.max(0, x);
    }

    // Flatten a 2D matrix into a 1D array
    public static double[] flatten(double[][] matrix) {
        return Arrays.stream(matrix)
                .flatMapToDouble(Arrays::stream)
                .toArray();
    }
}

// Convolutional Layer
class ConvolutionalLayer {
    private double[][] kernel;
    private int kernelSize;
    private double bias;

    public ConvolutionalLayer(int kernelSize) {
        this.kernelSize = kernelSize;
        this.kernel = new double[kernelSize][kernelSize];
        this.bias = Math.random() - 0.5;

        // Initialize kernel weights randomly
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                kernel[i][j] = Math.random() - 0.5;
            }
        }
    }

    // Perform convolution operation
    public double[][] forward(double[][] input) {
        int outputSize = input.length - kernelSize + 1;
        double[][] output = new double[outputSize][outputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double sum = 0;
                for (int ki = 0; ki < kernelSize; ki++) {
                    for (int kj = 0; kj < kernelSize; kj++) {
                        sum += input[i + ki][j + kj] * kernel[ki][kj];
                    }
                }
                output[i][j] = Utils.relu(sum + bias);
            }
        }

        return output;
    }
}

// Fully Connected Layer
class FullyConnectedLayer {
    private double[] weights;
    private double bias;

    public FullyConnectedLayer(int inputSize) {
        this.weights = new double[inputSize];
        this.bias = Math.random() - 0.5;

        // Initialize weights randomly
        for (int i = 0; i < inputSize; i++) {
            weights[i] = Math.random() - 0.5;
        }
    }

    // Perform forward pass
    public double forward(double[] input) {
        double sum = 0;
        for (int i = 0; i < input.length; i++) {
            sum += input[i] * weights[i];
        }
        return Utils.relu(sum + bias);
    }
}

// Simple CNN
class SimpleCNN {
    private ConvolutionalLayer convLayer;
    private FullyConnectedLayer fcLayer;

    public SimpleCNN(int kernelSize, int inputSize) {
        this.convLayer = new ConvolutionalLayer(kernelSize);
        this.fcLayer = new FullyConnectedLayer((inputSize - kernelSize + 1) * (inputSize - kernelSize + 1));
    }

    public double forward(double[][] input) {
        // Convolutional layer forward pass
        double[][] convOutput = convLayer.forward(input);

        // Flatten the convolutional layer output
        double[] flattenedOutput = Utils.flatten(convOutput);

        // Fully connected layer forward pass
        return fcLayer.forward(flattenedOutput);
    }
}

public class CNNExample {
    public static void main(String[] args) {
        // Input image (4x4)
        double[][] input = {
                {1, 2, 3, 0},
                {0, 1, 2, 3},
                {3, 0, 1, 2},
                {2, 3, 0, 1}
        };

        // Initialize a simple CNN
        int kernelSize = 2;
        int inputSize = 4;
        SimpleCNN cnn = new SimpleCNN(kernelSize, inputSize);

        // Forward pass
        double output = cnn.forward(input);
        System.out.printf("CNN Output: %.4f%n", output);
    }
}

