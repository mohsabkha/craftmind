package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.utils.MathUtils;
import com.craftsentient.craftmind.neuron.Neuron;
import lombok.*;

import java.util.*;
import java.util.stream.IntStream;

import static com.craftsentient.craftmind.activation.ActivationFunctions.activationFunction;
import static com.craftsentient.craftmind.utils.PrintUtils.print;

@Getter
@AllArgsConstructor
@Builder
public class DenseLayer {
    private ArrayList<Neuron> neuronList;
    private double[][] neuronWeights;
    private double[] neuronBiases;
    @Setter
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
    }

    public DenseLayer(double[][] weights, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) throws Exception {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.activationFunction = activationFunction;
        this.generateLayer(weights);
        this.generateLayerOutput();
    }

    public DenseLayer(double[][] weights, double[] inputs, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) throws Exception {
        this.neuronList = new ArrayList<>();
        this.neuronBiases = new double[0];
        this.layerOutputs = new double[0];
        this.activationFunction = activationFunction;
        this.neuronWeights = weights;
        this.inputs = inputs;
        this.generateLayer(weights, neuronBiases);
        this.generateLayerOutput();
    }

    public DenseLayer(double[][] weights, double[] biases, double[] inputs, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) throws Exception {
        this.neuronList = new ArrayList<>();
        this.layerOutputs = new double[0];
        this.neuronWeights = weights;
        this.neuronBiases = biases;
        this.inputs = inputs;
        this.activationFunction = activationFunction;
        this.generateLayer(this.neuronWeights, this.neuronBiases, this.inputs);
        this.generateLayerOutput();
    }

    public DenseLayer(int numberOfNeurons, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) throws Exception {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.activationFunction = activationFunction;
        this.generateLayer(numberOfNeurons);
        this.generateLayerOutput();
    }

    /**
     * calls either batched or non-batched layer output generation function
    */
    public double[] generateLayerOutput() throws Exception {
        if(this.inputs.length == 0){
            this.inputs = new double[this.getNeuronList().size()];
        }
        return generateLayerOutput(this.inputs);
    }

    public double[] regenerateLayerOutput(double[] inputs) {
        double[] newOutputs = new double[this.getLayerOutputs().length];
        IntStream.range(0, this.getLayerOutputs().length).forEachOrdered(i -> {
            this.setInputs(inputs);
            Neuron neuron = this.getNeuronList().get(i);
            newOutputs[i] = neuron.regenerateOutput(this.inputs);
        });
        this.layerOutputs = newOutputs;
        return this.layerOutputs;
    }

    public double[] regenerateLayerOutput() {
        double[] newOutputs = new double[this.getLayerOutputs().length];
        IntStream.range(0, this.getLayerOutputs().length).forEachOrdered(i -> {
            Neuron neuron = this.getNeuronList().get(i);
            newOutputs[i] = neuron.regenerateOutput(this.inputs);
        });
        this.layerOutputs = activationFunction(this.activationFunction,newOutputs);
        return this.layerOutputs;
    }

    /**
     * calls the generateNonBatchedLayerOutput for each entry in the batch of data, then calls the addOutput method to save the results
     * - Is called by the generateBatchedLayerOutput function
     */
    private double[] generateLayerOutput(double[] inputs) throws Exception {
        this.layerOutputs = new double[0];
        IntStream.range(0, this.neuronBiases.length).forEachOrdered(i -> {
            try { this.addOutput(Neuron.generateOutput(inputs, this.neuronWeights[i], this.neuronBiases[i])); }
            catch (Exception e) { throw new RuntimeException(e); }
        });
        this.layerOutputs = activationFunction(this.activationFunction, this.layerOutputs);
        Integer maxIndexOpt = IntStream.range(0, this.layerOutputs.length)
                .boxed()
                .max((i, j) -> Double.compare(this.layerOutputs[i], this.layerOutputs[j])).get();
        return this.layerOutputs;
    }


    public void addOutput(double value) throws Exception {
        this.layerOutputs = MathUtils.addToDoubleArray(this.layerOutputs, value);
    }

    public void addOutput(double value, double alpha, double beta) throws Exception {
        this.layerOutputs = MathUtils.addToDoubleArray(this.layerOutputs, value);
    }

    public void addNeuron(Neuron neuron) {
        this.neuronList.add(neuron);
        this.neuronWeights = MathUtils.addToDoubleArray(this.neuronWeights, neuron.getWeights());
        this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, neuron.getBias());
    }

    public void addNeurons(ArrayList<Neuron> neurons) {
        this.neuronList = neurons;
        IntStream.range(0, neurons.size()).parallel().forEachOrdered(i -> {
            this.neuronWeights = MathUtils.addToDoubleArray(this.neuronWeights, neurons.get(i).getWeights());
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, neurons.get(i).getBias());
        });
    }

    public void addWeights(double[][] weights) {
        this.neuronWeights = weights;
        IntStream.range(0, weights.length).parallel().forEachOrdered(i -> {
            this.neuronList.add(new Neuron(weights[i], 1));
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, 1);
        });
    }

    public void addWeightsAndBiases(double[][] weights, double[] biases) {
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
        this.inputs = MathUtils.addToDoubleArray(this.inputs, inputValue);
    }

    public void generateLayer(int numberOfNeurons) {
        this.neuronList = new ArrayList<>();
        for (int i = 0; i < numberOfNeurons; i++) {
            this.neuronList.add(new Neuron(numberOfNeurons, 1.0));
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, 1.0);
            this.inputs = MathUtils.addToDoubleArray(this.inputs, Neuron.randn());
        }
        this.addNeurons(this.neuronList);
    }

    public void generateLayer(double[][] weights) {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = weights;
        for (double[] weight : weights) {
            this.neuronList.add(new Neuron(weight, 1.0));
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, 1);
        }
    }

    public void generateLayer(double[][] weights, double[] biases) {
        this.neuronList = new ArrayList<>();
        if(biases.length == 0){
            biases = new double[weights.length];
        }
        this.neuronBiases = biases;
        for (int i = 0; i < weights.length; i++) {
            this.neuronList.add(new Neuron(weights[i], biases[i]));
        }
    }

    public void generateLayer(double[][] weights, double[] biases, double[] inputs) {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = weights;
        this.neuronBiases = biases;
        this.inputs = inputs;
        for (int i = 0; i < weights.length; i++) {
            this.neuronList.add(new Neuron(weights[i], biases[i]));
        }
    }
}