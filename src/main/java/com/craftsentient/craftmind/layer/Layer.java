package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.neuron.Neuron;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
@AllArgsConstructor
@Data
@Builder
public class Layer {
    private ArrayList<Neuron> neuronList;
    private ArrayList<Double> inputs;
    private ArrayList<Double> layerOutputs;

    public Layer(){
        this.neuronList = new ArrayList<>();
        this.inputs = new ArrayList<>();
        this.layerOutputs = new ArrayList<>();
    }

    public ArrayList<Double> generateLayerOutput(){

        this.layerOutputs = new ArrayList<>();
        neuronList.forEach(
                neuron -> {
                    layerOutputs.add(neuron.generateOutput(inputs));
                }
        );
        return this.layerOutputs;
    }

    public void addNeuron(Neuron neuron){
        this.neuronList.add(neuron);
    }

    public void addInput(Double inputValue){
        this.inputs.add(inputValue);
    }

    public void generateLayer(int numberOfNeurons){
        for(int i = 0; i <= numberOfNeurons; i++){
            this.neuronList.add(new Neuron(numberOfNeurons, 1.0));
        }
    }
}
