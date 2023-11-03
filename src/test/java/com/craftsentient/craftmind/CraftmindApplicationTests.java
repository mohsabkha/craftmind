package com.craftsentient.craftmind;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSS_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.ErrorLossFunctions;
import com.craftsentient.craftmind.layers.DenseLayers;
import com.craftsentient.craftmind.utils.PrintUtils;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.HashMap;

@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() throws Exception {
        double[][] inputs = {{1, 2, 3, 2.5}};
        double[][] weights = { // 1 per neuron
                {0.2, 0.8, -0.5, 1.0},
                {0.5, -0.91, 0.26, -0.5},
                {-0.26, -0.27, 0.17, 0.87}
        };
        double[] biases = {2, 3, 0.5}; // 1 per neuron

        DenseLayers builtLayerWithFile = new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(6)
                .withNumberOfNeuronsPerLayer(new int[]{3,5,6,7,4,3})
                .withInitialInput(inputs)
                .withInitialBiases(biases)
                .withInitialWeights(weights)
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.RELU_ACTIVATION_FUNCTION)
                .withLossFunction(DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION)
                .withTrueValue(new int[] {0, 1, 2})
                .build();

        builtLayerWithFile.generateDecisionsMap();
        PrintUtils.printInfo("decisions per batch", builtLayerWithFile.getDecisions());
        double[]loss = ErrorLossFunctions.lossFunction(DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION,
                new int[]{0,1,1}, builtLayerWithFile.getLayerAt(builtLayerWithFile.getLayerList().size()-1).getLayerOutputs());
    }

    @Test
    public void mainTest() {
        String [] args = new String[0];
        CraftmindApplication.main(args);
    }

}
