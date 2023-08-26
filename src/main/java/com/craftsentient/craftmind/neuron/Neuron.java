package com.craftsentient.craftmind.neuron;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
@Data
@NoArgsConstructor
@AllArgsConstructor
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
