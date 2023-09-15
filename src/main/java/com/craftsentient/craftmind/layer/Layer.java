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
    private double[] layerOutputs;

    public Layer() {
        this.neuronList = new ArrayList<>();
        this.inputs = new double[0];
        this.layerOutputs = new double[0];
    }

    // generating a list of output based on each neuron and the layer inputs
    public double[] generateLayerOutput() {
        this.layerOutputs = new double[0];
        IntStream.range(0, neuronList.size()).forEach(i -> {
            this.addOutput(neuronList.get(i).generateOutput(this.inputs));
        });
        return this.layerOutputs;

    }

    public void addOutput(double value){
        this.layerOutputs = MathUtils.addToDoubleArray(this.layerOutputs,  value);
    }


    // adding a neuron to the layers neuron
    public void addNeuron(Neuron neuron) {
        this.neuronList.add(neuron);
    }

    // adding an input value
    public void addInput(double inputValue) {
        this.inputs = MathUtils.addToDoubleArray(this.inputs, inputValue);
    }

    public void useOutputFromPreviousLayerAsInput(Layer layer){
        this.inputs = layer.generateLayerOutput();
    }

    // generating a layer with a vector of neurons
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
