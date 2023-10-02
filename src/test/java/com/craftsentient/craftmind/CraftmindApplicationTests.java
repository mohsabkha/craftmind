package com.craftsentient.craftmind;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSS_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.ErrorLossFunctions;
import com.craftsentient.craftmind.layers.DenseLayers;
import com.craftsentient.craftmind.utils.MathUtils;
import com.craftsentient.craftmind.utils.PrintUtils;
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
                .withNumberOfLayers(6)
                .withNumberOfNeuronsPerLayer(new int[]{3,5,6,7,4,3})
                .withInitialInput(inputs)
                .withInitialBiases(biases)
                .withInitialWeights(weights)
                .withSingleActivationFunctionForSingleLayer(0, DEFAULT_ACTIVATION_FUNCTIONS.RELU_ACTIVATION_FUNCTION)
                .withSingleActivationFunctionForSingleLayer(2, DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION)
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .withLossFunction(DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION)
                .withTrueValue(new int[] {0, 1, 2})
                .build();

        builtLayerWithFile.printLayers("BUILT WITH FILE NETWORK");
        PrintUtils.printInfo("Mean Loss", builtLayerWithFile.generateMeanLoss());
        builtLayerWithFile.batchDecisionsMap();
        PrintUtils.printInfo("decisions per batch", builtLayerWithFile.getDecisions());
        double[]loss = ErrorLossFunctions.lossFunction(DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION, new int[]{0,1,1}, new double[][]{{0.7,0.1,0.2}, {0.1,0.5,0.4}, {0.02,0.9,0.08}}, false);
        //PrintUtils.printPositive(MathUtils.mean(loss));
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
