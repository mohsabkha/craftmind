package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Getter;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
public interface Layer {

    public Object generateLayerOutput() throws Exception;

    public double[][] generateBatchedLayerOutput(int batchSize);

    public double[] generateNonBatchedLayerOutput(double[] inputs) throws Exception;

    public void generateOutput(int batchSize);

    public void addOutput(double value) throws Exception;

    public void addOutput(double[] values) throws Exception;

    public void addNeuron(Neuron neuron);

    public void addNeurons(ArrayList<Neuron> neurons);

    public void addWeights(double[][] weights);

    public void addWeightsAndBiases(double[][] weights, double[] biases);

    public void addInput(double inputValue);

    public void addInput(double[] inputsValues);

    public void addInput(double[][] inputValues);

    public void useOutputFromPreviousLayerAsInput(Layer layer) throws Exception;

    public void generateLayer(int numberOfNeurons);

    public void generateLayer(double[][] weights);

    public void generateLayer(double[][] weights, double[][] batchInputs);

    public void generateLayer(double[][] weights, double[] biases);

    public void generateLayer(double[][] weights, double[] biases, double[] inputs);

    public void generateLayer(double[][] weights, double[] biases, double[][] batchInputs);

    public Layer layerAddition(Layer a, Layer b);

    Layer layer(Layer a, Layer b);
}
