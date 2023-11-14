package com.craftsentient.craftmind.layers;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATIONS;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.craftsentient.craftmind.utils.PrintUtils.printPositive;

public class LayersTest {
    double[][] inputs;
    double[][] weights;
    double[] biases;
    @BeforeEach
    public void testAllConstructors() {
        inputs = new double[][]{
                {1, 2, 3, 2.5},
                {2, 5, -1, 2},
                {-1.5, 2.7, 3.3, -0.8}};
        weights = new double[][] {
                {0.2, 0.8, -0.5, 1.0},
                {0.5, -0.91, 0.26, -0.5},
                {-0.26, -0.27, 0.17, 0.87}};
        biases = new double[] {2, 3, 0.5};
    }

    @Test
    public void constructor1() {
        // this will auto create a classification neural network!
        printPositive("First DenseLayer");
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(5)
                .withTrueValueIndices(new int[]{4})
                .build().train();//.printLayers("NETWORK 1 NEW");
    }

    @Test
    public void constructor2() {
        // when no batch input is provided, one layer of input will be created
        printPositive("Second DenseLayer");
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(7)
                .withNumberOfNeurons(7)
                .withTrueValueIndices(new int[]{1})
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTPLUS_ACTIVATION_FUNCTION)
                .build().train();//.printLayers("NETWORK 2 NEW");
    }

    @Test
    public void constructor3() {
        // when no batch input is provided, one layer of input will be created with a specific amount of data
        printPositive("Third DenseLayer");
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(64)
                .withNumberOfInputs(27)
                .withNumberOfNeurons(64)
                .withTrueValueIndices(new int[]{0})
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build().train();//.printLayers("NETWORK 2A NEW");
    }

    @Test
    public void constructor4() {
        // create neural network with number of layers and specific number of neurons in each layer as input
        // with no input, it will default to the first layers number of neurons as batch size
        printPositive("Fourth Layer");
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(7)
                .withNumberOfNeuronsPerLayer(new int[]{2, 5, 9, 2, 5, 4, 10})
                .withHotOneVector(new int[][] {{0,0,0,0,0,0,0,1,0,0}})
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build().train();
    }

    @Test
    public void constructor5() {
        // create neural network with number of layers and initial inputs as input
        printPositive("Fifth Layer");
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(3)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{0,2,1})
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build()
                .train();
    }

    @Test
    public void constructor6() {
        // create neural network with number of layers, number of neurons per layer, and initial inputs as input
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(7)
                .withNumberOfNeurons(3)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{0,0,1})
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .withActivationFunctionForSingleLayer(3, DEFAULT_ACTIVATIONS.RELU_ACTIVATION_FUNCTION)
                .withActivationFunctionForMultipleLayers(5,6, DEFAULT_ACTIVATIONS.RELU_ACTIVATION_FUNCTION)
                .build()
                .train();
    }

    @Test
    public void constructor7() {
        // create neural network with number of layers, the specific number of neurons in each layer, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(3)
                .withNumberOfNeuronsPerLayer(new int[]{3, 5, 9})
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[] {2,1,1})
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .withAlpha(2.0)
                .withGamma(3.0)
                .withMargin(1.0)
                .withDelta(2.5)
                .build()
                .train();
    }

    @Test
    public void constructor8() {
        // create neural network with number of layers, initial weights for inputs, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(5)
                .withInitialWeights(weights)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{1,2,0})
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build()
                .train();
    }

    @Test
    public void constructor9() {
        // create neural network with number of layers, number of neurons per layer, initial weights for inputs, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(5)
                .withNumberOfNeurons(3)
                .withTrueValueIndices(new int[]{1,2,0})
                .withInitialWeights(weights)
                .withInitialInput(inputs)
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build()
                .train();
    }

    @Test
    public void constructor10() {
        // create neural network with number of layers, specific number of neurons per layer, initial weights for inputs, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(5)
                .withNumberOfNeuronsPerLayer( new int[]{3, 5, 9, 2, 7})
                .withInitialWeights(weights)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{0,2,0})
                .build()
                .train();
    }

    @Test
    public void constructor11() {
        // create neural network with number of layers, initial weights per input, biases for each neuron, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(4)
                .withInitialWeights(weights)
                .withInitialBiases(biases)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{1,0,0})
                //.withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build().train();//.printLayers("NETWORK 5A NEW");
    }

    @Test
    public void constructor12() {
        // create neural network with number of layers, number of neurons per layer, initial weights per input, biases for each neuron, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(4)
                .withNumberOfNeurons(3)
                .withInitialWeights(weights)
                .withInitialBiases(biases)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{1,2,2})
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build().train();//.printLayers("NETWORK 5B NEW");
    }

    @Test
    public void constructor13() {
        // create neural network with number of layers, specific number of neurons per layer, initial weights per input, biases for each neuron, and initial inputs
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(4)
                .withNumberOfNeuronsPerLayer(new int[]{3,5,6,7})
                .withInitialWeights(weights)
                .withInitialBiases(biases)
                .withInitialInput(inputs)
                .withTrueValueIndices(new int[]{1,2,0})
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build().train();//.printLayers("NETWORK 5C NEW");
    }

    @Test
    public void constructor14() {
        new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(4)
                .withNumberOfNeuronsPerLayer(new int[]{3,5,6,7})
                .withInitialWeights(weights)
                .withInitialBiases(biases)
                .withTextFileAsInput("src/main/resources/inputs.txt", ",")
                .withTrueValueIndices(new int[]{1,2,0})
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build().train();//.printLayers("NETWORK 5C NEW");

    }
}
