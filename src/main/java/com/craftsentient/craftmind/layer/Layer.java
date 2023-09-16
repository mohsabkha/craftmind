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
    private double[] inputs;
    private double[][] batchInputs;
    private double[] layerOutputs;
    private double[][] batchLayerOutputs;
    private boolean layerAsInput = false;

    public Layer() {
        this.neuronList = new ArrayList<>();
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
        this.batchInputs = new double[0][0];
        this.batchLayerOutputs = new double[0][0];
    }

    public Object generateLayerOutput(){
        if(this.inputs.length == 0){
            return generateLayerOutput(this.batchInputs.length);
        }
        return generateLayerOutput(this.inputs);
    }

    public double[] generateLayerOutput(double[] inputs) {
        this.layerOutputs = new double[0];
        IntStream.range(0, neuronList.size()).parallel().forEachOrdered(i -> {
            this.addOutput(neuronList.get(i).generateOutput(inputs));
        });
        return this.layerOutputs;
    }

    public double[][] generateLayerOutput(int batchSize){
        this.batchLayerOutputs = new double[0][0];
        IntStream.range(0, batchSize).parallel().forEachOrdered(i -> {
            this.generateLayerOutput(this.batchInputs[i]);
            this.addOutput(this.layerOutputs);
        });
        return this.batchLayerOutputs;
    }

    public void addOutput(double value){
        this.layerOutputs = MathUtils.addToDoubleArray(this.layerOutputs,  value);
    }
    public void addOutput(double[] values){
        this.batchLayerOutputs = MathUtils.addToDoubleArray(this.batchLayerOutputs,  values);
    }

    public void addNeuron(Neuron neuron) {
        this.neuronList.add(neuron);
    }

    public void addNeurons(ArrayList<Neuron> neurons){
        this.neuronList = neurons;
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
        this.layerAsInput = true;
        if(inputs.length != 0) this.inputs = (double[])layer.generateLayerOutput();
        else this.batchInputs = layer.getBatchLayerOutputs();
    }

    public Layer generateLayer(int numberOfNeurons) {
        for(int i = 0; i < numberOfNeurons; i++){
            this.neuronList.add(new Neuron(numberOfNeurons, 1.0));
            this.inputs = MathUtils.addToDoubleArray(this.inputs, Math.random() * ((1 - (-1)) + 1));
        }
        return this;
    }

    public static Layer addLayer(Layer a, Layer b){
        Neuron neuron = new Neuron();
        IntStream.range(0, a.getNeuronList().size()).parallel().forEach( i -> {
                    neuron.setOutput(a.getNeuronList().get(i).getOutput() + b.getNeuronList().get(i).getOutput());
                    a.getNeuronList().set(i, neuron);
                }
        );
        return a;
    }
}
