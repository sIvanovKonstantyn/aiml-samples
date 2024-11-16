package linear_regression;

import java.util.Arrays;

public class LinearRegression {

    private double[] weights; // Model weights
    private double bias;      // Bias term
    private double learningRate; // Learning rate
    private int epochs;          // Number of iterations

    public LinearRegression(int numFeatures, double learningRate, int epochs) {
        this.weights = new double[numFeatures];
        this.bias = 0.0;
        this.learningRate = learningRate;
        this.epochs = epochs;
    }

    // Mean Squared Error (MSE) function
    private double computeLoss(double[][] x, double[] y) {
        int m = x.length;
        double loss = 0.0;
        for (int i = 0; i < m; i++) {
            double prediction = predict(x[i]);
            loss += Math.pow(prediction - y[i], 2);
        }
        return loss / m;
    }

    // Predict the target value for a single data point
    // ------------LINEAR FUNCTION------------
    private double predict(double[] x) {
        double prediction = bias;
        for (int i = 0; i < weights.length; i++) {
            prediction += weights[i] * x[i];
        }
        return prediction;
    }

    // Train the model using gradient descent
    public void fit(double[][] x, double[] y) {
        int m = x.length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] gradientsW = new double[weights.length];
            double gradientB = 0.0;

            for (int i = 0; i < m; i++) {
                double prediction = predict(x[i]);
                double error = prediction - y[i];

                // Compute gradients for weights
                for (int j = 0; j < weights.length; j++) {
                    gradientsW[j] += error * x[i][j];
                }
                // Compute gradient for bias
                gradientB += error;
            }

            // Average gradients
            for (int j = 0; j < weights.length; j++) {
                gradientsW[j] /= m;
            }
            gradientB /= m;

            // Update weights and bias
            for (int j = 0; j < weights.length; j++) {
                weights[j] -= learningRate * gradientsW[j];
            }
            bias -= learningRate * gradientB;

            // Print loss every 100 epochs
            if (epoch % 100 == 0) {
                double loss = computeLoss(x, y);
                System.out.printf("Epoch %d: Loss = %.4f%n", epoch, loss);
            }
        }
    }

    // Predict the target values for multiple data points
    public double[] predict(double[][] X) {
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }

    // Get the trained weights (for debugging or analysis)
    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    public static void main(String[] args) {
        // Example data: features x and target values y
        double[][] x = {
                {75, 3, 5},
                {60, 2, 10},
                {100, 4, 2}
        };
        double[] y = {135000, 120000, 200000};

        // Create and train the model
        LinearRegression model = new LinearRegression(3, 0.0001, 1000);

        model.fit(x, y);

        // Print the trained parameters
        System.out.println("Trained weights: " + Arrays.toString(model.getWeights()));
        System.out.println("Trained bias: " + model.getBias());

        // Make predictions
        double[] predictions = model.predict(x);
        System.out.println("Predictions: " + Arrays.toString(predictions));
    }
}
