import java.io.*;
import java.util.*;

public class DiffusionModel {
    private Map<Integer, Set<Integer>> graph; // Adjacency list
    private Map<Integer, Boolean> activeNodes; // Activation status of each node
    private Map<Integer, Double> nodeAttributes; // Neural network features for nodes

    public DiffusionModel() {
        graph = new HashMap<>();
        activeNodes = new HashMap<>();
        nodeAttributes = new HashMap<>();
    }

    // Add an edge between two nodes
    public void addEdge(int node1, int node2) {
        graph.computeIfAbsent(node1, k -> new HashSet<>()).add(node2);
        graph.computeIfAbsent(node2, k -> new HashSet<>()).add(node1); // For undirected graph
    }

    // Load neural network outputs as node attributes
 public void loadNodeAttributes(String filePath) throws IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
        String line = br.readLine(); // Read the header line and discard it
        if (line != null) {
            line = br.readLine(); // Move to the next line containing data
        }
        
        // Now, read the data lines
        while (line != null) {
            String[] parts = line.split(",");
            int nodeId = Integer.parseInt(parts[0]);
            double featureValue = Double.parseDouble(parts[1]); // Assume single feature for simplicity
            nodeAttributes.put(nodeId, featureValue);
            line = br.readLine(); // Read the next line
        }
    }
}


    // Activate a node
    public void activateNode(int node) {
        activeNodes.put(node, true);
    }

    // Perform one step of diffusion
    public void diffuse(double baseThreshold) {
        Map<Integer, Boolean> newActiveNodes = new HashMap<>(activeNodes);

        for (Map.Entry<Integer, Set<Integer>> entry : graph.entrySet()) {
            int node = entry.getKey();
            Set<Integer> neighbors = entry.getValue();

            if (!activeNodes.getOrDefault(node, false)) {
                int activeNeighborCount = 0;
                double totalNeighborInfluence = 0.0;

                for (int neighbor : neighbors) {
                    if (activeNodes.getOrDefault(neighbor, false)) {
                        activeNeighborCount++;
                        totalNeighborInfluence += nodeAttributes.getOrDefault(neighbor, 1.0);
                    }
                }

                // Calculate activation threshold dynamically based on node attribute
                double threshold = baseThreshold / (1 + nodeAttributes.getOrDefault(node, 0.5));

                // Activate node if the influence from active neighbors exceeds threshold
                if (totalNeighborInfluence / neighbors.size() >= threshold) {
                    newActiveNodes.put(node, true);
                }
            }
        }

        activeNodes = newActiveNodes;
    }

    // Display the active nodes
    public void displayActiveNodes() {
        System.out.println("Active Nodes: ");
        activeNodes.forEach((node, isActive) -> {
            if (isActive) {
                System.out.print(node + " ");
            }
        });
        System.out.println();
    }

}
