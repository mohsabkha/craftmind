package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.mathUtils.MathUtils;
import com.craftsentient.craftmind.neuron.Neuron;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.stream.IntStream;

@Component
@AllArgsConstructor
@Data
@Builder
public class Layer {
    private ArrayList<Neuron> neuronList;
    private double[][] neuronWeights;
    private double[] neuronBiases;
    private double[] inputs;
    private double[][] batchInputs;
    private double[] layerOutputs;
    private double[][] batchLayerOutputs;
    private boolean isHiddenLayer = false;

    public Layer() {
        this.neuronList = new ArrayList<>();
        this.neuronWeights = new double[0][0];
        this.neuronBiases = new double[0];
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.batchInputs = new double[0][0];
        this.batchLayerOutputs = new double[0][0];
    }

    public Object generateLayerOutput(){
        if(this.inputs.length == 0){
            return generateBatchedLayerOutput(this.batchInputs.length);
        }
        return generateNonBatchedLayerOutput(this.inputs);
    }

    public double[][] generateBatchedLayerOutput(int batchSize){
        this.batchLayerOutputs = new double[0][0];
        if(this.isHiddenLayer){
            if(this.neuronWeights[0].length != this.batchInputs[0].length) { this.neuronWeights = MathUtils.transposeMatrix(this.neuronWeights); }
            this.generateOutput(this.batchInputs.length);
        } else {
            this.generateOutput(batchSize);
        }

        return this.batchLayerOutputs;
    }

    public double[] generateNonBatchedLayerOutput(double[] inputs) {
        this.layerOutputs = new double[0];
        IntStream.range(0, this.neuronBiases.length).forEach(i -> {
            this.addOutput(Neuron.generateOutput(inputs, this.neuronWeights[i], this.neuronBiases[i]));
        });
        return this.layerOutputs;
    }

    private void generateOutput(int batchSize){
        IntStream.range(0, batchSize).forEach(i -> {
            this.generateNonBatchedLayerOutput(this.batchInputs[i]);
            this.addOutput(this.layerOutputs);
        });
    }

    public void addOutput(double value){
        this.layerOutputs = MathUtils.addToDoubleArray(this.layerOutputs,  value);
    }

    public void addOutput(double[] values){
        this.batchLayerOutputs = MathUtils.addToDoubleArray(this.batchLayerOutputs,  values);
    }

    public void addNeuron(Neuron neuron) {
        this.neuronList.add(neuron);
        this.neuronWeights = MathUtils.addToDoubleArray(this.neuronWeights, neuron.getWeights());
        this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, neuron.getBias());
    }

    public void addNeurons(ArrayList<Neuron> neurons){
        this.neuronList = neurons;
        IntStream.range(0, neurons.size()).parallel().forEachOrdered(i -> {
            this.neuronWeights = MathUtils.addToDoubleArray(this.neuronWeights, neurons.get(i).getWeights());
            this.neuronBiases = MathUtils.addToDoubleArray(this.neuronBiases, neurons.get(i).getBias());
        });
    }

    public void addInput(double inputValue) {
        this.inputs = MathUtils.addToDoubleArray(this.inputs, inputValue);
    }

    public void addInput(double[] inputsValues){
        this.batchInputs = MathUtils.addToDoubleArray(this.batchInputs, inputsValues);
    }

    public void addInput(double[][] inputValues){
        IntStream.range(0, inputValues.length).forEach( i -> addInput(inputValues[i]));
    }

    public void useOutputFromPreviousLayerAsInput(Layer layer){
        this.isHiddenLayer = true;
        if(inputs.length != 0) this.inputs = (double[])layer.generateLayerOutput();
        else this.batchInputs = layer.getBatchLayerOutputs();
    }

    public Layer generateLayer(int numberOfNeurons) {
        for(int i = 0; i < numberOfNeurons; i++){
            this.neuronList.add(new Neuron(numberOfNeurons, 1.0));
            this.inputs = MathUtils.addToDoubleArray(this.inputs, Math.random() * ((1 - (-1)) + 1));
        }
        this.addNeurons(this.neuronList);
        return this;
    }

    public static Layer addLayers(Layer a, Layer b){
        Neuron neuron = new Neuron();
        IntStream.range(0, a.getNeuronList().size()).parallel().forEach( i -> {
                    neuron.setOutput(a.getNeuronList().get(i).getOutput() + b.getNeuronList().get(i).getOutput());
                    a.getNeuronList().set(i, neuron);
                }
        );
        return a;
    }
}
