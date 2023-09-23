package com.craftsentient.craftmind.layers;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
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

//        // create neural network with just number of layers as input
//        DenseLayers layers1 = new DenseLayers(5, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers1.printLayers("Layers 1");
//
//        // create neural network with number of layers and number of neurons per layer as input
//        DenseLayers layers2A = new DenseLayers(7, 7, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers2A.printLayers("Layers 2A");
//
//        // create neural network with number of layers and specific number of neurons in each layer as input
//        DenseLayers layers2B = new DenseLayers(7, new int[]{2, 5, 9, 2, 5, 4, 1}, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers2B.printLayers("Layers 2B");
//
//        // create neural network with number of layers and initial inputs as input
//        DenseLayers layers3A = new DenseLayers(3, inputs, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers3A.printLayers("Layers 3A");
//
//        // create neural network with number of layers, number of neurons per layer, and initial inputs as input
//        DenseLayers layers3B = new DenseLayers(5, 3, inputs,DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers3B.printLayers("Layers 3B");
//
//        // create neural network with number of layers, the specific number of neurons in each layer, and initial inputs
//        DenseLayers layers3C = new DenseLayers(3, new int[]{3, 5, 9, 4, 6, 3}, inputs, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers3C.printLayers("Layers 3C");
//
//        // create neural network with number of of layers, initial weights for inputs, and initial inputs
//        DenseLayers layers4A = new DenseLayers(5, weights, inputs, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers4A.printLayers("Layers 4A");
//
//        // create neural network with number of layers, number of neurons per layer, initial weights for inputs, and initial inputs
//        DenseLayers layers4B = new DenseLayers(5, 3, weights, inputs, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers4B.printLayers("Layers 4B");
//
//        // create neural network with number of layers, specific number of neurons per layer, initial weights for inputs, and initial inputs
//        DenseLayers layers4C = new DenseLayers(5, new int[]{3, 5, 9, 2, 7}, weights, inputs, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers4C.printLayers("Layers 4C");
//
//        // create neural network with number of layers, initial weights per input, biases for each neuron, and initial inputs
//        DenseLayers layers5A = new DenseLayers(4, weights, biases, inputs, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers5A.printLayers("Layers 5A");
//
//        // create neural network with number of layers, initial weights per input, biases for each neuron, and initial inputs
//        DenseLayers layers5B = new DenseLayers(4, 3, weights, biases, inputs, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION);
//        layers5B.printLayers("Layers 5B");

        // create neural network with number of layers, initial weights per input, biases for each neuron, and initial inputs
        DenseLayers layers5C = new DenseLayers(4, new int[]{3,5,6,7}, weights, biases, inputs,DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION);
        layers5C.printLayers("Layers 5C");

    }
}
