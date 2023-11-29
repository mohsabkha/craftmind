package com.craftsentient.craftmind.neuron;

import com.craftsentient.craftmind.utils.craftmath.MathUtils;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.util.stream.IntStream;

import static com.craftsentient.craftmind.layers.BaseNeuralNetwork.random;

@Component
@Data
@AllArgsConstructor
@Builder
public class Neuron {
    int size;
    private double output;
    private double bias;
    private double[] weights;
    private double[] inputs;

    public Neuron() {
        this.weights = new double[0];
        this.bias = 0.0;
        this.output = 0.0;
        this.size = 0;
    }

    public Neuron(int weights) {
        this.size = weights;
        this.bias = 0;
        this.weights = randn(weights);
    }

    public Neuron(int weights, double bias){
        this.size = weights;
        this.bias = bias;
        this.weights = randn(weights);
    }

    public Neuron(double[] weights, double bias){
        this.size = weights.length;
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

    public double regenerateOutput(double[] inputs){
        this.output = MathUtils.arrayDotProduct(inputs, this.weights) + this.bias;
        return this.output;
    }

    public void addWeight(double value){
        this.weights = MathUtils.addToDoubleArray(this.weights, value);
        this.size = weights.length;
    }

    public void setWeight(int index, double value){
        this.weights[index] = value;
    }

    public static double randn(){
        return 0.001 * random.nextGaussian();
    }

    public double[] randn(int inputs){
        double[] weights = new double[inputs];
        IntStream.range(0, inputs).parallel().forEachOrdered(i -> weights[i] = randn());
        return weights;
    }

}
