package com.craftsentient.craftmind;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSS_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.ErrorLossFunctions;
import com.craftsentient.craftmind.layers.DenseLayers;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.HashMap;

@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() throws Exception {
        double[][] inputs = {
                {1, 2, 3, 2.5},
                {2, 5, -1, 2},
                {-1.5, 2.7, 3.3, -0.8}};
        double[][] weights = {
                {0.2, 0.8, -0.5, 1.0},
                {0.5, -0.91, 0.26, -0.5},
                {-0.26, -0.27, 0.17, 0.87}};
        double[] biases = {2, 3, 0.5};

        DenseLayers builtLayerWithFile = new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(4)
                .withNumberOfNeuronsPerLayer(new int[]{3,5,6,7})
                .withInitialInput(inputs)
                .withInitialBiases(biases)
                .withInitialWeights(weights)
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .withSingleActivationFunctionForSingleLayer(0, DEFAULT_ACTIVATION_FUNCTIONS.RELU_ACTIVATION_FUNCTION)
                .build();

        DenseLayers builtLayer = new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(4)
                .withNumberOfNeuronsPerLayer(new int[]{3,5,6,7})
                .withTextFileAsInput("src/main/resources/inputs.txt", ",")
                .withInitialBiases(biases)
                .withInitialWeights(weights)
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .withSingleActivationFunctionForSingleLayer(0, DEFAULT_ACTIVATION_FUNCTIONS.RELU_ACTIVATION_FUNCTION)
                .build();


        builtLayer.printLayers("BUILD WITHOUT FILE");
        builtLayerWithFile.printLayers("BUILT WITH FILE NETWORK");

        double loss = ErrorLossFunctions.lossFunction(DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION, 0, new double[] {0.7, 0.1, 0.2});

        System.out.println(":::: LOSS FUNCTION :::: " + loss);


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
