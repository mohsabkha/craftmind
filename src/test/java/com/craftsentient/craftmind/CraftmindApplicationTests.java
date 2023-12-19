package com.craftsentient.craftmind;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATIONS;
import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSSES;
import com.craftsentient.craftmind.learningRate.DEFAULT_LEARNING_RATE;
import com.craftsentient.craftmind.neuralNetwork.BaseNeuralNetwork;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import static com.craftsentient.craftmind.testDataGenerator.DataGenerator.createData;


@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() throws Exception {
        double[][] inputs = {
                {1, 2, 3, 2.5},
                {2.0,7.0,-1.0,2.0},
                {-1.5,2.7,3.3,-0.8}
        };
        double[][] weights = { // 1 per neuron
                {0.2, 0.8, -0.5, 1.0},
                {0.5, -0.91, 0.26, -0.5},
                {-0.26, -0.27, 0.17, 0.87},
                {-0.5, 0.91, -0.26, 0.5}
        };
        double[] biases = {2, 3, 0.5, 1}; // 1 per neuron

        BaseNeuralNetwork builtLayerWithFile = new BaseNeuralNetwork.DenseLayersBuilder()
                .withNumberOfLayers(100)
                .withNumberOfNeuronsPerLayer(new int[]{4,12,3})
                .withInitialInput(inputs)
                .withInitialBiases(biases)
                .withInitialWeights(weights)
                .withEpoch(1)
                .withLearningRate(1)
                .withMiniBatchProcessing(1)
                .withMomentum(.9)
                .withLearningRateFunction(DEFAULT_LEARNING_RATE.RMSPROP)
                .withActivationFunction(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .withLossFunction(DEFAULT_LOSSES.CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION)
                .withTrueValueIndices(new int[] {0, 1, 2})
                .build();
        builtLayerWithFile.train();
    }

    @Test
    public void mainTest() {
        String [] args = new String[0];
        CraftmindApplication.main(args);
    }

    @Test
    public void testWithSpiralData(){
        // Example usage
        double[][] X; // x y coordinates
        int[] y; // true values
        int samples = 1000; // Number of samples per class
        int classes = 3;   // Number of classes

        Object[] data = createData(samples, classes);
        X = (double[][]) data[0];
        y = (int[]) data[1];

        double[][] weights = { // 1 per neuron
                {0.2, 0.8},
                {2.0, -0.91}
        };
        double[] biases = {0.1, 0.1}; // 1 per neuron

        BaseNeuralNetwork builtLayerWithFile = new BaseNeuralNetwork.DenseLayersBuilder()
                .withNumberOfLayers(3)
                .withEpoch(10)
                .withNumberOfNeuronsPerLayer(new int[]{2,64,3})
                .withInitialInput(X)
                .withInitialBiases(biases)
                .withInitialWeights(weights)
                .withLearningRate(0.001)
                .withMiniBatchProcessing(1)
                .withMomentum(0.9)
                .withLearningRateFunction(DEFAULT_LEARNING_RATE.MOMENTUM)
                .withActivationFunction(DEFAULT_ACTIVATIONS.RELU_6_ACTIVATION_FUNCTION)
                .withActivationFunctionForOutput(DEFAULT_ACTIVATIONS.SOFTMAX_ACTIVATION_FUNCTION)
                .withLossFunction(DEFAULT_LOSSES.CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION)
                .withTrueValueIndices(y)
                .build();
        builtLayerWithFile.train();
    }
}
