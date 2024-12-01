import java.io.*;
import java.util.*;

public class DataPreprocessor {
    // Updated method to load line graph data from a CSV file without input size
    public static List<double[]> loadLineGraphData(String filePath) throws IOException {
        List<double[]> lineGraphData = new ArrayList<>();
        int inputSize = 0;  // This will be determined from the first row

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            // Read the header line and determine the number of columns
            String headerLine = br.readLine();
            if (headerLine == null) {
                throw new IOException("The CSV file is empty.");
            }
            String[] headers = headerLine.split(",");
            inputSize = headers.length;  // Set the input size based on the number of columns in the header

            // Read the rest of the file line by line
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length != inputSize) {
                    throw new IOException("Data row does not match the expected number of columns.");
                }
                double[] graphPoint = new double[inputSize];
                for (int i = 0; i < inputSize; i++) {
                    graphPoint[i] = Double.parseDouble(values[i]);
                }
                lineGraphData.add(graphPoint);
            }
        }
        return lineGraphData;
    }
}
