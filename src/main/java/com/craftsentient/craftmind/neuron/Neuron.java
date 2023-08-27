package com.craftsentient.craftmind.neuron;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Random;
import java.util.random.RandomGenerator;

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
        this.bias = bias;
        this.weights = new ArrayList<>();
        for(int i = 0; i < numberOfWeights; i++){
            weights.add(Math.random() * ((max - min) + min));
        }
    }

    public Double generateOutput(ArrayList<Double>inputs){
        for(int i = 0; i < inputs.size(); i++){
            this.output += (inputs.get(i) * this.weights.get(i));
        }
        this.output += this.bias;
        return this.output;
    }
}
