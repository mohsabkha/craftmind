package com.craftsentient.craftmind.neuron;

import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;

@Data
@Builder
public class Neuron {
    private Long output;
    private Long bias;
    private ArrayList<Long> inputs;
    private ArrayList<Long> weights;

    public Long generateOutput(){
        for(int i = 0; i < this.inputs.size(); i++){
            this.output += (this.inputs.get(i) * this.weights.get(i));
        }
        this.output += this.bias;
        return this.output;
    }
}
