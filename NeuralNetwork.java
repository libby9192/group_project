import java.io.*;
import java.util.*;

public class NeuralNetwork {
    private double[][] weightsInputToHidden;
    private double[][] weightsHiddenToOutput;
    private double[] hiddenLayerBias;
    private double[] outputLayerBias;

    private int inputSize;
    private int hiddenSize;
    private int outputSize;

    // Member variables to store intermediate layer outputs
    private double[] hiddenLayerInput;
    private double[] outputLayerInput;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // Initialize weights and biases with random values
        Random random = new Random();
        weightsInputToHidden = new double[inputSize][hiddenSize];
        weightsHiddenToOutput = new double[hiddenSize][outputSize];
        hiddenLayerBias = new double[hiddenSize];
        outputLayerBias = new double[outputSize];

        // Random initialization of weights and biases
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputToHidden[i][j] = random.nextDouble() - 0.5;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsHiddenToOutput[i][j] = random.nextDouble() - 0.5;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            hiddenLayerBias[i] = random.nextDouble() - 0.5;
        }

        for (int i = 0; i < outputSize; i++) {
            outputLayerBias[i] = random.nextDouble() - 0.5;
        }
    }

    // Method to process input data and save the output to a file
    public void processAndSave(List<double[]> inputData, String outputFilePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
            // Write header (optional, based on output format)
            writer.write("NodeID, FeatureValue");
            writer.newLine();

            // Process each input data point through the neural network
            for (int i = 0; i < inputData.size(); i++) {
                double[] output = forward(inputData.get(i));  // Get neural network output for the data point
                writer.write(i + "," + output[0]);  // Assuming one output per input
                writer.newLine();
            }
        }
    }

    // Forward pass through the network
    public double[] forward(double[] input) {
        hiddenLayerInput = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                hiddenLayerInput[i] += input[j] * weightsInputToHidden[j][i];
            }
            hiddenLayerInput[i] += hiddenLayerBias[i];
            hiddenLayerInput[i] = relu(hiddenLayerInput[i]);  // ReLU activation
        }

        outputLayerInput = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                outputLayerInput[i] += hiddenLayerInput[j] * weightsHiddenToOutput[j][i];
            }
            outputLayerInput[i] += outputLayerBias[i];
            outputLayerInput[i] = sigmoid(outputLayerInput[i]);  // Sigmoid activation
        }

        return outputLayerInput;
    }

    // Backpropagation method to adjust weights and biases
    public void backpropagate(double[] input, double[] errors, double learningRate) {
        // Calculate the gradients for the output layer
        double[] outputGradients = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputGradients[i] = errors[i] * sigmoidDerivative(outputLayerInput[i]);
        }

        // Calculate the gradients for the hidden layer
        double[] hiddenGradients = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            hiddenGradients[i] = 0;
            for (int j = 0; j < outputSize; j++) {
                hiddenGradients[i] += outputGradients[j] * weightsHiddenToOutput[i][j];
            }
            hiddenGradients[i] *= reluDerivative(hiddenLayerInput[i]);
        }

        // Update weights and biases for the output layer
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsHiddenToOutput[i][j] += learningRate * outputGradients[j] * hiddenLayerInput[i];
            }
        }
        // Update biases for the output layer
        for (int i = 0; i < outputSize; i++) {
            outputLayerBias[i] += learningRate * outputGradients[i];
        }

        // Update weights and biases for the hidden layer
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputToHidden[i][j] += learningRate * hiddenGradients[j] * input[i];
            }
        }
        // Update biases for the hidden layer
        for (int i = 0; i < hiddenSize; i++) {
            hiddenLayerBias[i] += learningRate * hiddenGradients[i];
        }
    }

    // Derivative of ReLU activation function
    private double reluDerivative(double x) {
        return (x > 0) ? 1 : 0;
    }

    // Derivative of Sigmoid activation function
    private double sigmoidDerivative(double x) {
        return x * (1 - x);  // Sigmoid derivative
    }

    private double relu(double x) {
        return Math.max(0, x);
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
