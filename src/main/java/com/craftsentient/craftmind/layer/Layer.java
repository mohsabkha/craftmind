package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Getter;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
public interface Layer {

    public void addOutput(double value) throws Exception;

    public void addNeurons(ArrayList<Neuron> neurons);

    public void addInput(double[] inputsValues);

    public void useOutputFromPreviousLayerAsInput(Layer layer) throws Exception;

    public void generateLayer(int numberOfNeurons);

    public void generateLayer(double[][] weights);

    public void generateLayer(double[][] weights, double[][] batchInputs);

    public void generateLayer(double[][] weights, double[] biases);

    public void generateLayer(double[][] weights, double[] biases, double[] inputs);

    public void generateLayer(double[][] weights, double[] biases, double[][] batchInputs);

    public Layer layerAddition(Layer a, Layer b);
}
