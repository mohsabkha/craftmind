package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.activation.ActivationFunctions;
import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;
import com.craftsentient.craftmind.mathUtils.MathUtils;
import com.craftsentient.craftmind.neuron.Neuron;
import lombok.*;

import java.util.ArrayList;
import java.util.stream.IntStream;

import static com.craftsentient.craftmind.activation.ActivationFunctions.activationFunction;

@Getter
@AllArgsConstructor
@Builder
public class DenseLayer implements Layer {
    private ArrayList<Neuron> neuronList;
    private double[][] neuronWeights;
    private double[] neuronBiases;
    private double[] inputs;
    private double[][] batchInputs;
    @Setter
    private double[] layerOutputs;
    @Setter
    private double[][] batchLayerOutputs;
    private boolean isHiddenLayer = true;
    @Setter
    private DEFAULT_ACTIVATION_FUNCTIONS activationFunction;

    public DenseLayer() {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.batchInputs = new double[0][0];
        this.batchLayerOutputs = new double[0][0];
        this.activationFunction = DEFAULT_ACTIVATION_FUNCTIONS.LINEAR_ACTIVATION_FUNCTION;
    }

    public DenseLayer(double[][] weights, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.batchInputs = new double[0][0];
        this.batchLayerOutputs = new double[0][0];
        this.activationFunction = activationFunction;
        this.generateLayer(weights);
        this.generateBatchedLayerOutput(weights.length);
    }

    public DenseLayer(double[][] weights, double[] biases, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.batchInputs = new double[0][0];
        this.batchLayerOutputs = new double[0][0];
        this.activationFunction = activationFunction;
        this.generateLayer(weights, biases);
        this.generateBatchedLayerOutput(weights.length);
    }

    public DenseLayer(double[][] weights, double[][] batchInputs, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.batchInputs = new double[0][0];
        this.batchLayerOutputs = new double[0][0];
        this.activationFunction = activationFunction;
        this.generateLayer(weights, batchInputs);
        this.generateBatchedLayerOutput(weights.length);
    }

    public DenseLayer(double[][] weights, double[] biases, double[] inputs, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.batchInputs = new double[0][0];
        this.batchLayerOutputs = new double[0][0];
        this.activationFunction = activationFunction;
        this.generateLayer(weights, biases, inputs);
        this.generateBatchedLayerOutput(weights.length);
    }

    public DenseLayer(double[][] weights, double[] biases, double[][] batchInputs, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.batchInputs = new double[0][0];
        this.batchLayerOutputs = new double[0][0];
        this.activationFunction = activationFunction;
        this.generateLayer(weights, biases, batchInputs);
        this.generateBatchedLayerOutput(weights.length);
    }

    public DenseLayer(int numberOfNeurons, DEFAULT_ACTIVATION_FUNCTIONS activationFunction) {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.batchInputs = new double[0][0];
        this.batchLayerOutputs = new double[0][0];
        this.activationFunction = activationFunction;
        this.generateLayer(numberOfNeurons);
        this.generateBatchedLayerOutput(numberOfNeurons);
    }

    public Object generateLayerOutput() throws Exception {
        if (this.inputs.length == 0) {
            return generateBatchedLayerOutput(this.batchInputs.length);
        }
        return generateNonBatchedLayerOutput(this.inputs);
    }

    public double[][] generateBatchedLayerOutput(int batchSize) {
        this.batchLayerOutputs = new double[0][0];
        if (this.isHiddenLayer) {
            if (this.neuronWeights[0].length != this.batchInputs[0].length) {
                this.neuronWeights = MathUtils.transposeMatrix(this.neuronWeights);
            }
            this.generateOutput(this.batchInputs.length);
        } else {
            this.generateOutput(batchSize);
        }

        return this.batchLayerOutputs;
    }

    public double[] generateNonBatchedLayerOutput(double[] inputs) throws Exception {
        this.layerOutputs = new double[0];
        IntStream.range(0, this.neuronBiases.length).forEach(i -> {
            try {
                this.addOutput(Neuron.generateOutput(inputs, this.neuronWeights[i], this.neuronBiases[i]));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        this.layerOutputs = activationFunction(this.activationFunction, this.layerOutputs);
        return this.layerOutputs;
    }

    public void generateOutput(int batchSize) {
        IntStream.range(0, batchSize).forEach(i -> {
            try {
                this.generateNonBatchedLayerOutput(this.batchInputs[i]);
                this.addOutput(this.layerOutputs);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    public void addOutput(double value) throws Exception {
        this.layerOutputs = MathUtils.addToDoubleArray(this.layerOutputs, value);
    }

    public void addOutput(double[] values) throws Exception {
        this.batchLayerOutputs = MathUtils.addToDoubleArray(this.batchLayerOutputs, values);
    }

    public void addOutput(double value, double alpha, double beta) throws Exception {
        this.layerOutputs = MathUtils.addToDoubleArray(this.layerOutputs, value);
    }

    public void addOutput(double[] values, double[] alphas, double[] betas) throws Exception {
        this.batchLayerOutputs = MathUtils.addToDoubleArray(this.batchLayerOutputs,  activationFunction(this.activationFunction, values, alphas, betas));
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

    public void addInput(double[] inputsValues) {
        this.batchInputs = MathUtils.addToDoubleArray(this.batchInputs, inputsValues);
    }

    public void addInput(double[][] inputValues) {
        IntStream.range(0, inputValues.length).forEach(i -> addInput(inputValues[i]));
    }

    @Override
    public void useOutputFromPreviousLayerAsInput(Layer layer) throws Exception {
        useOutputFromPreviousLayerAsInput((DenseLayer) layer);
    }

    public void useOutputFromPreviousLayerAsInput(DenseLayer layer) throws Exception {
        this.isHiddenLayer = true;
        if (inputs.length != 0) this.inputs = (double[]) layer.generateLayerOutput();
        else this.batchInputs = layer.getBatchLayerOutputs();
    }

    public void generateLayer(int numberOfNeurons) {
        for (int i = 0; i < numberOfNeurons; i++) {
            this.neuronList.add(new Neuron(numberOfNeurons, 1.0));
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, 1.0);
            this.inputs = MathUtils.addToDoubleArray(this.inputs, Neuron.randn());
        }
        this.addNeurons(this.neuronList);
    }

    public void generateLayer(double[][] weights) {
        this.neuronWeights = weights;
        for (double[] weight : weights) {
            this.neuronList.add(new Neuron(weight, 1.0));
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, 1);
        }
    }

    public void generateLayer(double[][] weights, double[][] batchInputs) {
        this.neuronWeights = weights;
        this.batchInputs = batchInputs;
        for (double[] weight : weights) {
            double bias = Neuron.randn();
            this.neuronList.add(new Neuron(weight, bias));
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, bias);
        }
    }

    public void generateLayer(double[][] weights, double[] biases) {
        this.neuronWeights = weights;
        this.neuronBiases = biases;
        for (int i = 0; i < weights.length; i++) {
            this.neuronList.add(new Neuron(weights[i], biases[i]));
        }
    }

    public void generateLayer(double[][] weights, double[] biases, double[] inputs) {
        this.neuronWeights = weights;
        this.neuronBiases = biases;
        this.inputs = inputs;
        for (int i = 0; i < weights.length; i++) {
            this.neuronList.add(new Neuron(weights[i], biases[i]));
        }
    }

    public void generateLayer(double[][] weights, double[] biases, double[][] batchInputs) {
        this.neuronWeights = weights;
        this.neuronBiases = biases;
        this.batchInputs = batchInputs;
        for (int i = 0; i < weights.length; i++) {
            this.neuronList.add(new Neuron(weights[i], biases[i]));
        }
    }

    @Override
    public Layer layerAddition(Layer a, Layer b) {
        return layerAddition(a, b);
    }

    @Override
    public Layer layer(Layer a, Layer b) {
        return layerAddition(a, b);
    }

    public static DenseLayer layerAddition(DenseLayer a, DenseLayer b) {
        Neuron neuron = new Neuron();
        IntStream.range(0, a.getNeuronList().size()).parallel().forEach(i -> {
            neuron.setOutput(a.getNeuronList().get(i).getOutput() + b.getNeuronList().get(i).getOutput());
            a.getNeuronList().set(i, neuron);
        });
        return a;
    }

    public Neuron getNeuronAt(int index) {
        return this.getNeuronList().get(index);
    }

}