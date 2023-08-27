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
    private ArrayList<Double> layerOutput;

    public Layer(){
        this.neuronList = new ArrayList<>();
        this.inputs = new ArrayList<>();
        this.layerOutput = new ArrayList<>();
    }

    public ArrayList<Double> generateLayerOutput(){

        this.layerOutput = new ArrayList<>();
        neuronList.forEach(
                neuron -> {
                    layerOutput.add(neuron.generateOutput(inputs));
                }
        );
        return this.layerOutput;
    }

    public void addNeuron(Neuron neuron){
        this.neuronList.add(neuron);
    }

    public void addInput(Double inputValue){
        this.inputs.add(inputValue);
    }
}
