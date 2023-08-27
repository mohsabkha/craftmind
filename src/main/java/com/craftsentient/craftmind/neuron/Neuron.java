package com.craftsentient.craftmind.neuron;

import com.craftsentient.craftmind.utils.MathUtils;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
@Data
@AllArgsConstructor
@Builder
public class Neuron {
    int max;
    int min;
    private Double output;
    private Double bias;
    private ArrayList<Double> weights;

    public Neuron(){
        this.weights = new ArrayList<>();
        this.bias = 0.0;
        this.output = 0.0;
        this.min = 0;
        this.max = 0;
    }

    public Neuron(int numberOfWeights, Double bias){
        int maximum = 1;
        int minimum = -1;
        this.bias = bias;
        this.weights = new ArrayList<>();
        for(int i = 0; i < numberOfWeights; i++){
            weights.add(Math.random() * ((maximum - minimum) + minimum));
        }
    }

    public Neuron(int numberOfWeights, Double bias, int max, int min){
        this.max = max;
        this.min = min;
        this.bias = bias;
        this.weights = new ArrayList<>();
        for(int i = 0; i < numberOfWeights; i++){
            weights.add(Math.random() * ((max - min) + min));
        }
    }

    public Double generateOutput(ArrayList<Double>inputs){
        this.output = MathUtils.dotProduct(inputs, this.weights) + this.bias;
        return this.output;
    }
}
