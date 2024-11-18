package rnn;

import java.util.Arrays;

// Utility class for common operations
class Utils {
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double sig = sigmoid(x);
        return sig * (1 - sig);
    }
}

// RNN Cell
class RNNCell {
    private double[][] weightsInput;
    private double[][] weightsHidden;
    private double[] bias;
    private double[] hiddenState;

    public RNNCell(int inputSize, int hiddenSize) {
        weightsInput = new double[inputSize][hiddenSize];
        weightsHidden = new double[hiddenSize][hiddenSize];
        bias = new double[hiddenSize];
        hiddenState = new double[hiddenSize];

        // Initialize weights and biases randomly
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInput[i][j] = Math.random() - 0.5;
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsHidden[i][j] = Math.random() - 0.5;
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            bias[i] = Math.random() - 0.5;
        }
    }

    // Forward pass for a single time step
    public double[] forward(double[] input) {
        double[] newHiddenState = new double[hiddenState.length];

        // Compute new hidden state: h_t = sigmoid(Wx * x_t + Wh * h_(t-1) + b)
        for (int i = 0; i < newHiddenState.length; i++) {
            double sum = bias[i];
            for (int j = 0; j < input.length; j++) {
                sum += input[j] * weightsInput[j][i];
            }
            for (int j = 0; j < hiddenState.length; j++) {
                sum += hiddenState[j] * weightsHidden[j][i];
            }
            newHiddenState[i] = Utils.sigmoid(sum);
        }

        // Update hidden state
        hiddenState = newHiddenState;
        return hiddenState;
    }

    // Reset hidden state (optional for new sequences)
    public void resetHiddenState() {
        Arrays.fill(hiddenState, 0);
    }
}

// Simple RNN
class SimpleRNN {
    private RNNCell rnnCell;

    public SimpleRNN(int inputSize, int hiddenSize) {
        this.rnnCell = new RNNCell(inputSize, hiddenSize);
    }

    // Forward pass through the RNN for a sequence
    public double[][] forward(double[][] inputSequence) {
        double[][] outputs = new double[inputSequence.length][];
        for (int t = 0; t < inputSequence.length; t++) {
            outputs[t] = rnnCell.forward(inputSequence[t]);
        }
        return outputs;
    }

    // Reset hidden state
    public void reset() {
        rnnCell.resetHiddenState();
    }
}

public class RNNExample {
    public static void main(String[] args) {
        // Example input: Sequence of length 3, each element has 2 features
        double[][] inputSequence = {
                {0.5, 0.1},
                {0.3, 0.7},
                {0.6, 0.9}
        };

        // Initialize RNN: input size = 2, hidden size = 3
        int inputSize = 2;
        int hiddenSize = 3;
        SimpleRNN rnn = new SimpleRNN(inputSize, hiddenSize);

        // Forward pass
        double[][] outputSequence = rnn.forward(inputSequence);

        // Print outputs
        System.out.println("RNN Outputs:");
        for (double[] output : outputSequence) {
            System.out.println(Arrays.toString(output));
        }
    }
}
