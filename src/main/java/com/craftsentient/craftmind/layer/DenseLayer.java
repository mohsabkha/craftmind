package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSS_FUNCTIONS;
import com.craftsentient.craftmind.errorLoss.ErrorLossFunctions;
import com.craftsentient.craftmind.utils.MathUtils;
import com.craftsentient.craftmind.neuron.Neuron;
import lombok.*;

import java.util.*;
import java.util.stream.IntStream;

import static com.craftsentient.craftmind.activation.ActivationFunctions.activationFunction;
import static com.craftsentient.craftmind.utils.PrintUtils.printInfo;

@Getter
@AllArgsConstructor
@Builder
public class DenseLayer {
    private ArrayList<Neuron> neuronList;
    private double[][] neuronWeights;
    private double[] neuronBiases;
    private double[] inputs;
    private boolean isHiddenLayer = true;
    private double alpha;
    private double beta;
    private double delta;
    private double sigma;
    private double outputMean;
    @Setter
    private double[] layerOutputs;
    @Setter
    private DEFAULT_ACTIVATION_FUNCTIONS activationFunction;

    public DenseLayer() {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.activationFunction = DEFAULT_ACTIVATION_FUNCTIONS.SOFTMAX_ACTIVATION_FUNCTION;
        printInfo("Values set in DenseLayer()");
    }

    public DenseLayer(double[][] weights, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) throws Exception {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.activationFunction = activationFunction;
        printInfo("Values set in DenseLayer(double[][] weights, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] batchHotOneVec, int[] batchTrueValue)");
        printInfo("Beginning generation of individual layer and layer outputs...");
        this.generateLayer(weights);
        this.generateLayerOutput();
        printInfo("Generation of individual layer and layer outputs complete!");
    }

    public DenseLayer(double[][] weights, double[] biases, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) throws Exception {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.activationFunction = activationFunction;
        printInfo("Values set in DenseLayer(double[][] weights, double[] biases, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] batchHotOneVec, int[] batchTrueValue)");
        printInfo("Beginning generation of individual layer and layer outputs...");
        this.generateLayer(weights, biases);
        this.generateLayerOutput();
        printInfo("Generation of individual layer and layer outputs complete!");
    }

    public DenseLayer(double[][] weights, double[] biases, double[] inputs, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) throws Exception {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = inputs;
        this.layerOutputs = new double[0];
        this.activationFunction = activationFunction;
        printInfo("Values set in DenseLayer(double[][] weights, double[] biases, double[] inputs, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] batchHotOneVec, int[] batchTrueValue)");
        printInfo("Beginning generation of individual layer and layer outputs...");
        this.generateLayer(weights, biases, inputs);
        this.generateLayerOutput();
        printInfo("Generation of individual layer and layer outputs complete!");
    }

    public DenseLayer(int numberOfNeurons, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) throws Exception {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.activationFunction = activationFunction;
        printInfo("Values set in DenseLayer(int numberOfNeurons, DEFAULT_ACTIVATION_FUNCTIONS activationFunction, DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] batchHotOneVec, int[] batchTrueValue)");
        printInfo("Beginning generation of individual layer and layer outputs...");
        this.generateLayer(numberOfNeurons);
        this.generateLayerOutput();
        printInfo("Generation of individual layer and layer outputs complete!");
    }

    /**
     * calls either batched or non-batched layer output generation function
    */
    public Object generateLayerOutput() throws Exception {
        printInfo("Entered generateLayerOutput()");
        return generateLayerOutput(this.inputs);
    }

    /**
     * calls the generateNonBatchedLayerOutput for each entry in the batch of data, then calls the addOutput method to save the results
     * - Is called by the generateBatchedLayerOutput function
     */
    public double[] generateLayerOutput(double[] inputs) throws Exception {
        printInfo("Entered generateNonBatchedLayerOutput(double[] inputs)");
        this.layerOutputs = new double[0];
        IntStream.range(0, this.neuronBiases.length).forEach(i -> {
            try {
                this.addInput(inputs[i]);
                this.addOutput(Neuron.generateOutput(inputs, this.neuronWeights[i], this.neuronBiases[i]));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        this.layerOutputs = activationFunction(this.activationFunction, this.layerOutputs);

        Integer maxIndexOpt = IntStream.range(0, this.layerOutputs.length)
                .boxed()
                .max((i, j) -> Double.compare(this.layerOutputs[i], this.layerOutputs[j])).get();
        return this.layerOutputs;
    }


    public void addOutput(double value) throws Exception {
        printInfo("Entered addOutput(double value)");
        this.layerOutputs = MathUtils.addToDoubleArray(this.layerOutputs, value);
    }

    public void addOutput(double value, double alpha, double beta) throws Exception {
        printInfo("Entered addOutput(double value, double alpha, double beta)");
        this.layerOutputs = MathUtils.addToDoubleArray(this.layerOutputs, value);
    }

    public void addNeuron(Neuron neuron) {
        printInfo("Entered addNeuron(Neuron neuron)");
        this.neuronList.add(neuron);
        this.neuronWeights = MathUtils.addToDoubleArray(this.neuronWeights, neuron.getWeights());
        this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, neuron.getBias());
    }

    public void addNeurons(ArrayList<Neuron> neurons) {
        printInfo("Entered addNeurons(ArrayList<Neuron> neurons)");
        this.neuronList = neurons;
        IntStream.range(0, neurons.size()).parallel().forEachOrdered(i -> {
            this.neuronWeights = MathUtils.addToDoubleArray(this.neuronWeights, neurons.get(i).getWeights());
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, neurons.get(i).getBias());
        });
    }

    public void addWeights(double[][] weights) {
        printInfo("Entered addWeights(double[][] weights)");
        this.neuronWeights = weights;
        IntStream.range(0, weights.length).parallel().forEachOrdered(i -> {
            this.neuronList.add(new Neuron(weights[i], 1));
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, 1);
        });
    }

    public void addWeightsAndBiases(double[][] weights, double[] biases) {
        printInfo("Entered addWeightsAndBiases(double[][] weights, double[] biases)");
        this.neuronWeights = weights;
        this.neuronBiases = biases;
        IntStream.range(0, weights.length).parallel().forEachOrdered(i -> {
            this.neuronList.add(new Neuron(weights[i], biases[i]));
        });
        if (this.neuronList.isEmpty()) {
            this.generateLayer(weights, biases);
        }
    }

    public void addInput(double inputValue) {
        printInfo("Entered addInput(double inputValue)");
        this.inputs = MathUtils.addToDoubleArray(this.inputs, inputValue);
    }

    public void generateLayer(int numberOfNeurons) {
        printInfo("Entered generateLayer(int numberOfNeurons)");
        this.neuronList = new ArrayList<>();
        for (int i = 0; i < numberOfNeurons; i++) {
            this.neuronList.add(new Neuron(numberOfNeurons, 1.0));
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, 1.0);
            this.inputs = MathUtils.addToDoubleArray(this.inputs, Neuron.randn());
        }
        this.addNeurons(this.neuronList);
    }

    public void generateLayer(double[][] weights) {
        printInfo("Entered generateLayer(double[][] weights)");
        this.neuronList = new ArrayList<>();
        this.neuronWeights = weights;
        for (double[] weight : weights) {
            this.neuronList.add(new Neuron(weight, 1.0));
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, 1);
        }
    }

    public void generateLayer(double[][] weights, double[] biases) {
        printInfo("Entered generateLayer(double[][] weights, double[] biases)");
        this.neuronList = new ArrayList<>();
        this.neuronWeights = weights;
        this.neuronBiases = biases;
        for (int i = 0; i < weights.length; i++) {
            this.neuronList.add(new Neuron(weights[i], biases[i]));
        }
    }

    public void generateLayer(double[][] weights, double[] biases, double[] inputs) {
        printInfo("Entered generateLayer(double[][] weights, double[] biases, double[] inputs)");
        this.neuronList = new ArrayList<>();
        this.neuronWeights = weights;
        this.neuronBiases = biases;
        this.inputs = inputs;
        printInfo("Adding neurons to neuron list...");
        for (int i = 0; i < weights.length; i++) {
            this.neuronList.add(new Neuron(weights[i], biases[i]));
        }
    }
}