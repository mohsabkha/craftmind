package com.craftsentient.craftmind;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSS_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.ErrorLossFunctions;
import com.craftsentient.craftmind.layers.DenseLayers;
import com.craftsentient.craftmind.utils.PrintUtils;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;


@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() throws Exception {
        double[][] inputs = {
                {1, 2, 3, 2.5},
                {2.0,7.0,-1.0,2.0},
                {-1.5,2.7,3.3,-0.8},
                {1, 2, 10, 2.5},
                {2.0,5.0,-1.67,2.0},
                {-1.5,2.73,3.3,-0.8},
                {1, 2, 10, 2.5},
                {2.0,5.0,-1.67,2.0},
                {-1.5,2.73,3.3,-0.8}
        };
        double[][] weights = { // 1 per neuron
                {0.2, 0.8, -0.5, 1.0},
                {0.5, -0.91, 0.26, -0.5},
                {-0.26, -0.27, 0.17, 0.87}
        };
        double[] biases = {2, 3, 0.5}; // 1 per neuron

        DenseLayers builtLayerWithFile = new DenseLayers.DenseLayersBuilder()
                .withNumberOfLayers(6)
                .withNumberOfNeuronsPerLayer(new int[]{3,5,6,7,4,10})
                .withInitialInput(inputs)
                .withInitialBiases(biases)
                .withInitialWeights(weights)
                .withActivationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .withLossFunction(DEFAULT_LOSS_FUNCTIONS.NLL_LOSS_FUNCTION)
                .withTrueValueIndices(new int[] {1, 1, 2, 1, 1, 0, 2, 1, 1})
                .build();
        builtLayerWithFile.train();
    }

    @Test
    public void mainTest() {
        String [] args = new String[0];
        CraftmindApplication.main(args);
    }

}
