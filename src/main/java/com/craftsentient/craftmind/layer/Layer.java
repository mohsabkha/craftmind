package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Getter;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
public interface Layer {
    @Getter
    ArrayList<Neuron> neuronList = null;
    @Getter
    double[][] neuronWeights = new double[0][0];
    @Getter
    double[] neuronBiases = new double[0];
    @Getter
    double[] inputs = new double[0];
    @Getter
    double[][] batchInputs = new double[0][0];
    @Getter
    double[] layerOutputs = new double[0];
    @Getter
    double[][] batchLayerOutputs = new double[0][0];
    @Getter
    boolean isHiddenLayer = true;

    public Object generateLayerOutput();

    public double[][] generateBatchedLayerOutput(int batchSize);

    public double[] generateNonBatchedLayerOutput(double[] inputs);

    public void generateOutput(int batchSize);

    public void addOutput(double value);

    public void addOutput(double[] values);

    public void addNeuron(Neuron neuron);

    public void addNeurons(ArrayList<Neuron> neurons);

    public void addWeights(double[][] weights);

    public void addWeightsAndBiases(double[][] weights, double[] biases);

    public void addInput(double inputValue);

    public void addInput(double[] inputsValues);

    public void addInput(double[][] inputValues);

    public void useOutputFromPreviousLayerAsInput(Layer layer);

    public void generateLayer(int numberOfNeurons);

    public void generateLayer(double[][] weights);

    public void generateLayer(double[][] weights, double[][] batchInputs);

    public void generateLayer(double[][] weights, double[] biases);

    public void generateLayer(double[][] weights, double[] biases, double[] inputs);


    public void generateLayer(double[][] weights, double[] biases, double[][] batchInputs);

    public Layer addLayers(Layer a, Layer b);

}
