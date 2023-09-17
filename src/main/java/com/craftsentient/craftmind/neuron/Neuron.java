package com.craftsentient.craftmind.neuron;

import com.craftsentient.craftmind.mathUtils.MathUtils;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.util.stream.IntStream;

@Component
@Data
@AllArgsConstructor
@Builder
public class Neuron {
    int size;
    int max;
    int min;
    private double output;
    private double bias;
    private double[] weights;

    public Neuron() {
        this.weights = new double[0];
        this.bias = 0.0;
        this.output = 0.0;
        this.min = 0;
        this.max = 0;
        this.size = 0;
    }

    public Neuron(int weights, double bias) {
        this.size = 0;
        this.max = 1;
        this.min = -1;
        this.bias = bias;
        this.weights = IntStream.range(0, weights).parallel().mapToDouble(i -> Math.random() * ((this.max - this.min) + this.min)).toArray();
    }

    public Neuron(int weights, double bias, int max, int min){
        this.max = max;
        this.min = min;
        this.bias = bias;
        this.weights = IntStream.range(0, weights).parallel().mapToDouble(i -> Math.random() * ((this.max - this.min) + this.min)).toArray();
    }

    public Neuron(double[] weights, double bias){
        this.bias = bias;
        this.weights = weights;
    }

    public double generateOutput(double[] inputs){
        this.output = MathUtils.arrayDotProduct(inputs, this.weights) + this.bias;
        return this.output;
    }

    public static double generateOutput(double[] inputs, double[] weights, double bias){
        return MathUtils.arrayDotProduct(inputs, weights) + bias;
    }

    public void addWeight(double value){
        this.weights = MathUtils.addToDoubleArray(this.weights, value);
    }

}
