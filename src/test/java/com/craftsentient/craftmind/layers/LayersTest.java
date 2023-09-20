package com.craftsentient.craftmind.layers;

import org.junit.jupiter.api.Test;

public class LayersTest {
    @Test
    public void testAllConstructors() {

        double[][] inputs = {
                {1, 2, 3, 2.5},
                {2, 5, -1, 2},
                {-1.5, 2.7, 3.3, -0.8}};
        double[][] weights = {
                {0.2, 0.8, -0.5, 1.0},
                {0.5, -0.91, 0.26, -0.5},
                {-0.26, -0.27, 0.17, 0.87}};
        double[] biases = {2, 3, 0.5};

        // create neural network with just number of layers as input
        DenseLayers layers1 = new DenseLayers(5);
        layers1.printLayers("Layer 1");

        // create neural network with number of layers and number of neurons per layer as input
        DenseLayers layers2A = new DenseLayers(7, 7);
        layers2A.printLayers("Layer 2A");

        // create neural network with number of layers and specific number of neurons in each layer as input
        DenseLayers layers2B = new DenseLayers(7, new int[]{2, 5, 9, 2, 5, 4, 1});
        layers2B.printLayers("Layer 2B");

        // create neural network with number of layers and initial inputs as input
        DenseLayers layers3A = new DenseLayers(3, inputs);
        layers3A.printLayers("Layer 3A");

        // create neural network with number of layers, number of neurons per layer, and initial inputs as input
        DenseLayers layers3B = new DenseLayers(5, 3, inputs);
        layers3B.printLayers("Layer 3B");

        // create neural network with number of layers, the specific number of neurons in each layer, and initial inputs
        DenseLayers layers3C = new DenseLayers(3, new int[]{3, 5, 9, 4, 6, 3}, inputs);
        layers3C.printLayers("Layer 3C");

        // create neural network with number of of layers, initial weights for inputs, and initial inputs
        DenseLayers layers4A = new DenseLayers(5, weights, inputs);
        layers4A.printLayers("Layer 4A");

        // create neural network with number of layers, number of neurons per layer, initial weights for inputs, and initial inputs
        DenseLayers layers4B = new DenseLayers(5, 3, weights, inputs);
        layers4B.printLayers("Layer 4B");

        // create neural network with number of layers, specific number of neurons per layer, initial weights for inputs, and initial inputs
        DenseLayers layers4C = new DenseLayers(5, new int[]{3, 5, 9, 2, 7}, weights, inputs);
        layers4C.printLayers("Layer 4C");

        // create neural network with number of layers, initial weights per input, biases for each neuron, and initial inputs
        DenseLayers layers5A = new DenseLayers(4, weights, biases, inputs);
        layers5A.printLayers("Layer 5A");

        // create neural network with number of layers, initial weights per input, biases for each neuron, and initial inputs
        DenseLayers layers5B = new DenseLayers(4, 3, weights, biases, inputs);
        layers5B.printLayers("Layer 5B");

        // create neural network with number of layers, initial weights per input, biases for each neuron, and initial inputs
        DenseLayers layers5C = new DenseLayers(4, new int[]{3,5,6,7}, weights, biases, inputs);
        layers5C.printLayers("Layer 5C");

    }
}
