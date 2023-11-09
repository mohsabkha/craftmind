package com.craftsentient.craftmind.layers;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import org.junit.jupiter.api.Test;

import java.util.HashMap;

import static com.craftsentient.craftmind.utils.PrintUtils.printInfo;
import static com.craftsentient.craftmind.utils.PrintUtils.printPositive;

public class LayersTest {
    @Test
    public void testAllConstructors() throws Exception {

        double[][] inputs = {
                {1, 2, 3, 2.5},
                {2, 5, -1, 2},
                {-1.5, 2.7, 3.3, -0.8}};
        double[][] weights = {
                {0.2, 0.8, -0.5, 1.0},
                {0.5, -0.91, 0.26, -0.5},
                {-0.26, -0.27, 0.17, 0.87}};
        double[] biases = {2, 3, 0.5};

        // this will auto create a classification neural network!
        printPositive("First DenseLayer");
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(5)
                .withTrueValueIndices(new int[]{4})
                .build().train();//.printLayers("NETWORK 1 NEW");

        // when no batch input is provided, one layer of input will be created
        printPositive("Second DenseLayer");
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(7)
                .withNumberOfNeurons(7)
                .withTrueValueIndices(new int[]{1})
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTPLUS_ACTIVATION_FUNCTION)
                .build();//.printLayers("NETWORK 2 NEW");

        // when no batch input is provided, one layer of input will be created with a specific amount of data
        printPositive("Third DenseLayer");
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(64)
                .withNumberOfInputs(27)
                .withNumberOfNeurons(64)
                .withTrueValueIndices(new int[]{0})
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build().train();//.printLayers("NETWORK 2A NEW");

        // create neural network with number of layers and specific number of neurons in each layer as input
        // with no input, it will default to the first layers number of neurons as batch size
        printPositive("Fourth Layer");
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(7)
                .withNumberOfNeuronsPerLayer(new int[]{2, 5, 9, 2, 5, 4, 10})
                .withTrueValueIndices(new int[]{8})
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build().train();//.printLayers("NETWORK 2B NEW");

        // create neural network with number of layers and initial inputs as input
        printPositive("Fifth Layer");
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(3)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{0,2,1})
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build();//.printLayers("NETWORK 3A NEW");

        // create neural network with number of layers, number of neurons per layer, and initial inputs as input
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(7)
                .withNumberOfNeurons(3)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{0,0,1})
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build();//.printLayers("NETWORK 3B NEW");

        // create neural network with number of layers, the specific number of neurons in each layer, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(3)
                .withNumberOfNeuronsPerLayer(new int[]{3, 5, 9})
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[] {2,1,1})
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build();//.printLayers("NETWORK 3C NEW");

        // create neural network with number of layers, initial weights for inputs, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(5)
                .withInitialWeights(weights)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{1,2,0})
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build();//.printLayers("NETWORK 4A NEW");

        // create neural network with number of layers, number of neurons per layer, initial weights for inputs, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(5)
                .withNumberOfNeurons(3)
                .withTrueValueIndices(new int[]{1,2,0})
                .withInitialWeights(weights)
                .withInitialInput(inputs)
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build();//.printLayers("NETWORK 4B NEW");

        // create neural network with number of layers, specific number of neurons per layer, initial weights for inputs, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(5)
                .withNumberOfNeuronsPerLayer( new int[]{3, 5, 9, 2, 7})
                .withInitialWeights(weights)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{0,2,0})
                .build();

        // create neural network with number of layers, initial weights per input, biases for each neuron, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(4)
                .withInitialWeights(weights)
                .withInitialBiases(biases)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{1,0,0})
                //.withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build();//.printLayers("NETWORK 5A NEW");

        // create neural network with number of layers, number of neurons per layer, initial weights per input, biases for each neuron, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(4)
                .withNumberOfNeurons(3)
                .withInitialWeights(weights)
                .withInitialBiases(biases)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{1,2,2})
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build();//.printLayers("NETWORK 5B NEW");

        // create neural network with number of layers, specific number of neurons per layer, initial weights per input, biases for each neuron, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(4)
                .withNumberOfNeuronsPerLayer(new int[]{3,5,6,7})
                .withInitialWeights(weights)
                .withInitialBiases(biases)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{1,2,0})
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build();//.printLayers("NETWORK 5C NEW");
    }
}
