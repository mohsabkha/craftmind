package com.craftsentient.craftmind.neuron;

import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;

@Data
@Builder
public class NeuronImpl implements  Neuron {
    ArrayList<Long> inputs;
    ArrayList<Long> weights;
    Long bias;
    Long output;

    public Long createOutput(){
        long op = 0L;
        for(int i = 0; i < this.inputs.size(); i++){
            op += this.inputs.get(i) * this.weights.get(i);
        }
        op += this.bias;
        this.output = op;
        return this.output;
    }

    public Long createOutput(ArrayList<Long> inputs, ArrayList<Long> weights, Long bias){
        long op = 0L;
        for(int i = 0; i < inputs.size(); i++){
            op += inputs.get(i) * weights.get(i);
        }
        op += bias;
        this.output = op;
        return this.output;
    }

    public static Neuron createOutput(NeuronImpl neuron){
        long op = 0L;
        for(int i = 0; i < neuron.inputs.size(); i++){
            op += neuron.inputs.get(i) * neuron.weights.get(i);
        }
        op += neuron.bias;
        neuron.output = op;
        return neuron;
    }

}
