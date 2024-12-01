import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Prompt for input and output file paths
        System.out.print("Enter the CSV file path for input data: ");
        String inputFilePath = scanner.nextLine();

        System.out.print("Enter the output file path for neural network results: ");
        String neuralNetworkOutputFilePath = scanner.nextLine();

        System.out.print("Enter the output file path for diffusion model results: ");
        String diffusionModelOutputFilePath = scanner.nextLine();

        try {
            // Load data and process
            List<double[]> inputData = DataPreprocessor.loadLineGraphData(inputFilePath);

            // Determine the input size from the data (number of features per row)
            int inputSize = inputData.get(0).length;
            int hiddenSize = 5; // You can adjust this based on your model design
            int outputSize = 1; // Adjust as necessary for your use case

            // Initialize Neural Network with the determined input size
            NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSize, outputSize);

            // Initialize ModelTrainerValidator and train the model
            ModelTrainerValidator trainer = new ModelTrainerValidator(nn);
            trainer.train(inputData, 100, 0.01);  // Example: Train for 100 epochs with a learning rate of 0.01

            // Validate the model (you can pass a separate test dataset here)
            double accuracy = trainer.validate(inputData);
            System.out.println("Model accuracy: " + accuracy);

            // Process data and save output
            nn.processAndSave(inputData, neuralNetworkOutputFilePath);

            // Initialize Diffusion Model and load the neural network output
            DiffusionModel dm = new DiffusionModel();
            dm.loadNodeAttributes(neuralNetworkOutputFilePath);

            // Run the diffusion model
            double baseThreshold = 0.5; // Adjust threshold based on your application
            dm.diffuse(baseThreshold);

            // Save the diffusion model output
            dm.displayActiveNodes();  // Optionally display active nodes
            // Implement saving the diffusion model results if needed

            // Initialize ApplicationInterface to request input data and display predictions
            ApplicationInterface appInterface = new ApplicationInterface();
            appInterface.requestInputDataAndDisplay(nn);

        } catch (IOException e) {
            System.err.println("Error processing the data: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("An error occurred: " + e.getMessage());
        }
    }
}
