package com.craftsentient.craftmind.neuron;

import com.craftsentient.craftmind.utils.MathUtils;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.util.stream.IntStream;

import static com.craftsentient.craftmind.layers.DenseLayers.random;
import static com.craftsentient.craftmind.utils.PrintUtils.printInfo;

@Component
@Data
@AllArgsConstructor
@Builder
public class Neuron {
    int size;
    private double output;
    private double bias;
    private double[] weights;

    public Neuron() {
        this.weights = new double[0];
        this.bias = 0.0;
        this.output = 0.0;
        this.size = 0;
        printInfo("Values set in neuron with no input...");
    }

    public Neuron(int weights) {
        this.size = weights;
        this.bias = 0;
        this.weights = randn(weights);
        printInfo("Weights of neuron randomly generated...");
        printInfo("Values set in neuron with number of weights...");
    }

    public Neuron(int weights, double bias){
        this.size = weights;
        this.bias = bias;
        this.weights = randn(weights);
        printInfo("Weights of neuron randomly generated...");
        printInfo("Values set in neuron with number of weights and biases...");

    }

    public Neuron(double[] weights, double bias){
        this.size = weights.length;
        this.bias = bias;
        this.weights = weights;
        printInfo("Values set in neuron with inputted weights and number of biases...");
    }

    public double generateOutput(double[] inputs){
        printInfo("Entered generateOutput(double[] inputs) - Generating neuron output...");
        this.output = MathUtils.arrayDotProduct(inputs, this.weights) + this.bias;
        return this.output;
    }

    public static double generateOutput(double[] inputs, double[] weights, double bias){
        printInfo("Entered generateOutput(double[] inputs, double[] weights, double bias) - Generating neuron output...");
        return MathUtils.arrayDotProduct(inputs, weights) + bias;
    }

    public void addWeight(double value){
        printInfo("Entered addWeight(double value) - Adding weight value to neuron...");
        this.weights = MathUtils.addToDoubleArray(this.weights, value);
        this.size = weights.length;
    }

    public static double randn(){
        return 0.01 * random.nextGaussian();
    }

    public double[] randn(int inputs){
        double[] weights = new double[inputs];
        IntStream.range(0, inputs).parallel().forEachOrdered(i -> weights[i] = randn());
        return weights;
    }

}
