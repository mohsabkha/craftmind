package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.neuron.Neuron;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Component
@AllArgsConstructor
@Data
@Builder
public class Layer {

    private ArrayList<Neuron> neuronList;
    private ArrayList<Double> inputs;
    private ArrayList<Double> layerOutputs;

    public Layer() {
        this.neuronList = new ArrayList<>();
        this.inputs = new ArrayList<>();
        this.layerOutputs = new ArrayList<>();
    }

    // generating a list of output based on each neuron and the layer inputs
    public ArrayList<Double> generateLayerOutput() {
        this.layerOutputs = new ArrayList<>();
        neuronList.forEach( neuron -> { layerOutputs.add((Double)neuron.generateOutput(inputs)); });
        return this.layerOutputs;
    }


    // adding a neuron to the layers neuron
    public void addNeuron(Neuron neuron) {
        this.neuronList.add(neuron);
    }

    // adding an input value
    public void addInput(Double inputValue) {
        this.inputs.add(inputValue);
    }

    // generating a layer with a vector of neurons
    public Layer generateLayer(int numberOfNeurons) {
        for(int i = 0; i < numberOfNeurons; i++){
            this.neuronList.add(new Neuron(numberOfNeurons, 1.0));
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
