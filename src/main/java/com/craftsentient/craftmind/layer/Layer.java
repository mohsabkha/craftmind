package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.neuron.NeuronImpl;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class Layer {
    private List<NeuronImpl> neuronList;
    private ArrayList<Long> inputs;
    private ArrayList<Long> weights;
    private Long preBiasOutput;

    public Long createOutput(){
        this.preBiasOutput = 0L;
        for(int i = 0; i < this.inputs.size(); i++){
            this.preBiasOutput += this.inputs.get(i) * this.weights.get(i);
        }
        return this.preBiasOutput;
    }
}
