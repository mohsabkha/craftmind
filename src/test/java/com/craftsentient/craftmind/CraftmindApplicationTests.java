package com.craftsentient.craftmind;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.layers.DenseLayers;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.HashMap;

@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() {
        double[][] inputs = {
                {1, 2, 3, 2.5},
                {2, 5, -1, 2},
                {-1.5, 2.7, 3.3, -0.8}};
        double[][] weights = {
                {0.2, 0.8, -0.5, 1.0},
                {0.5, -0.91, 0.26, -0.5},
                {-0.26, -0.27, 0.17, 0.87}};
        double[] biases = {2, 3, 0.5};

        DenseLayers builtLayer = new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(4)
                .withNumberOfNeuronsPerLayer(new int[]{3,5,6,7})
                .withInitialInput(inputs)
                .withInitialBiases(biases)
                .withInitialWeights(weights)
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION)
                .withSingleActivationFunctionForSingleLayer(2, DEFAULT_ACTIVATION_FUNCTIONS.GAUSSIAN_ACTIVATION_FUNCTION)
                .withSingleActivationFunctionForMultipleLayers(0,1, DEFAULT_ACTIVATION_FUNCTIONS.RELU_ACTIVATION_FUNCTION)
                .withSingleActivationFunctionForSingleLayer(1, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION)
                .withSingleActivationFunctionForSingleLayer(3, DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .build();
        builtLayer.printLayers("BUILT NETWORK");
        DenseLayers layers5C = new DenseLayers(4, new int[]{3,5,6,7}, weights, biases, inputs,DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION, new HashMap<>());
        layers5C.printLayers("Layers 5C");


    }

    @Test
    public void mainTest() {
        String [] args = new String[0];
        CraftmindApplication.main(args);
    }

    @Test
    public void matrixTest(){

    }
}
