import java.io.*;
import java.util.*;

public class ApplicationInterface {

    // Displaying model results (simplified)
    public void displayPredictions(List<double[]> predictions) {
        System.out.println("Predictions:");
        for (double[] prediction : predictions) {
            System.out.println("Prediction: " + prediction[0]);
        }
    }

    // Optionally, you can add functionality to allow users to input new data for predictions
    public void requestInputDataAndDisplay(NeuralNetwork nn) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the number of data points to predict: ");
        int dataPoints = scanner.nextInt();
        scanner.nextLine();  // Consume newline character

        List<double[]> inputData = new ArrayList<>();
        for (int i = 0; i < dataPoints; i++) {
            System.out.print("Enter values for data point " + (i + 1) + ": ");
            String line = scanner.nextLine();
            String[] values = line.split(",");
            double[] input = new double[values.length];
            for (int j = 0; j < values.length; j++) {
                input[j] = Double.parseDouble(values[j]);
            }
            inputData.add(input);
        }

        // Display predictions
        List<double[]> predictions = new ArrayList<>();
        for (double[] data : inputData) {
            double[] result = nn.forward(data);
            predictions.add(result);
        }
        displayPredictions(predictions);
    }
}
