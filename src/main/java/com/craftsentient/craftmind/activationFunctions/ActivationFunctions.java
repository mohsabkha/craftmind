package com.craftsentient.craftmind.activationFunctions;

import java.util.Arrays;
import java.util.stream.IntStream;

public class ActivationFunctions {
    public static double activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double value, double beta) throws Exception {
        switch (activationFunction) {
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> {
                return exponentialElu(value);
            }
            case GAUSSIAN_ACTIVATION_FUNCTION -> {
                return gaussian(value);
            }
            case LEAKY_RELU_ACTIVATION_FUNCTION -> {
                return leakyRelu(value);
            }
            case LINEAR_ACTIVATION_FUNCTION -> {
                return linear(value);
            }
            case MISH_ACTIVATION_FUNCTION -> {
                return mish(value);
            }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> {
                return parametricRelu(value);
            }
            case RELU_ACTIVATION_FUNCTION -> {
                return rectifiedLinearUnit(value);
            }
            case SIGMOID_ACTIVATION_FUNCTION -> {
                return sigmoid(value);
            }
            case SOFTPLUS_ACTIVATION_FUNCTION -> {
                return softplus(value);
            }
            case SWISH_ACTIVATION_FUNCTION -> {
                return swish(value, beta);
            }
            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(value);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }

    public static double[] activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double[] values, double[] betas) throws Exception {
        switch (activationFunction) {
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> {
                return exponentialElu(values);
            }
            case GAUSSIAN_ACTIVATION_FUNCTION -> {
                return gaussian(values);
            }
            case LEAKY_RELU_ACTIVATION_FUNCTION -> {
                return leakyRelu(values);
            }
            case LINEAR_ACTIVATION_FUNCTION -> {
                return linear(values);
            }
            case MISH_ACTIVATION_FUNCTION -> {
                return mish(values);
            }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> {
                return parametricRelu(values);
            }
            case RELU_ACTIVATION_FUNCTION -> {
                return rectifiedLinearUnit(values);
            }
            case SIGMOID_ACTIVATION_FUNCTION -> {
                return sigmoid(values);
            }
            case SOFTMAX_ACTIVATION_FUNCTION -> {
                return softmax(values);
            }
            case SOFTPLUS_ACTIVATION_FUNCTION -> {
                return softplus(values);
            }
            case SWISH_ACTIVATION_FUNCTION -> {
                return swish(values, betas);
            }
            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(values);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());

        }
    }

    public static double[][] activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double[][] values, double[][] betas) throws Exception {
        switch (activationFunction) {
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> {
                return exponentialElu(values);
            }
            case GAUSSIAN_ACTIVATION_FUNCTION -> {
                return gaussian(values);
            }
            case LEAKY_RELU_ACTIVATION_FUNCTION -> {
                return leakyRelu(values);
            }
            case LINEAR_ACTIVATION_FUNCTION -> {
                return linear(values);
            }
            case MISH_ACTIVATION_FUNCTION -> {
                return mish(values);
            }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> {
                return parametricRelu(values);
            }
            case RELU_ACTIVATION_FUNCTION -> {
                return rectifiedLinearUnit(values);
            }
            case SIGMOID_ACTIVATION_FUNCTION -> {
                return sigmoid(values);
            }
            case SOFTMAX_ACTIVATION_FUNCTION -> {
                return softmax(values);
            }
            case SOFTPLUS_ACTIVATION_FUNCTION -> {
                return softplus(values);
            }
            case SWISH_ACTIVATION_FUNCTION -> {
                return swish(values, betas);
            }
            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(values);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());

        }
    }

    // EXPONENTIAL_ELU
    private static double exponentialElu(double value){
        return value;
    }
    private static double[] exponentialElu(double[] values){
        return values;
    }
    private static double[][] exponentialElu(double[][] values){
        return values;
    }


    // GAUSSIAN_ACTIVATION
    private static double gaussian(double value){
        return Math.exp(-(value*value));
    }
    private static double[] gaussian(double[] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> {
            values[i] = gaussian(values[i]);
        });
        return values;
    }
    private static double[][] gaussian(double[][] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> {
            values[i] = gaussian(values[i]);
        });
        return values;
    }


    // LEAKY_RELU
    private static double leakyRelu(double value){
        return value;
    }
    private static double[] leakyRelu(double[] values){
        return values;
    }
    private static double[][] leakyRelu(double[][] values){
        return values;
    }


    // LINEAR_ACTIVATION
    private static double linear(double value){
        return value;
    }
    private static double[] linear(double[] values){
        return values;
    }
    private static double[][] linear(double[][] values){
        return values;
    }


    // MISH_ACTIVATION
    private static double mish(double value){
        return value;
    }
    private static double[] mish(double[] values){
        return values;
    }
    private static double[][] mish(double[][] values){
        return values;
    }


    // PARAMETRIC_RELU
    private static double parametricRelu(double value){
        return value;
    }
    private static double[] parametricRelu(double[] values){
        return values;
    }
    private static double[][] parametricRelu(double[][] values){
        return values;
    }


    // RELU_ACTIVATION
    private static double rectifiedLinearUnit(double value){
        return Math.max(0, value);
    }
    private static double[] rectifiedLinearUnit(double[] values){
        return Arrays.stream(values).map(ActivationFunctions::rectifiedLinearUnit).toArray();
    }
    private static double[][] rectifiedLinearUnit(double[][] values){
        for(int i = 0; i < values.length; i++) {
            values[i] = rectifiedLinearUnit(values[i]);
        }
        return values;
    }


    // SIGMOID_ACTIVATION
    private static double sigmoid(double value){
        return 1.0 / (1.0 + Math.exp(-value));
    }
    private static double[]sigmoid(double[] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = sigmoid(values[i]));
        return values;
    }
    private static double[][] sigmoid(double[][] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = sigmoid(values[i]));
        return values;
    }

    private static double[] softmax(double[] values){
        double sum = 0.0;
        for (double value : values) { sum += Math.exp(value); }
        for (int i = 0; i < values.length; i++) { values[i] = Math.exp(values[i]) / sum; }
        return values;
    }
    private static double[][] softmax(double[][] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = softmax(values[i]));
        return values;
    }


    // SOFTPLUS
    private static double softplus(double value){
        return value;
    }
    private static double[] softplus(double[] values){
        return values;
    }
    private static double[][] softplus(double[][] values){
        return values;
    }


    // SWISH
    private static double swish(double value, double beta){
        return value * sigmoid(value);
    }
    private static double[] swish(double[] values, double[] beta){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = values[i]*sigmoid(values[i]));
        return values;
    }
    private static double[][] swish(double[][] values, double[][] beta){
        return values;
    }


    // TANH
    private static double tanh(double value){
        return (2/(1 + Math.exp(-(2*value)))) - 1;
    }
    private static double[] tanh(double[] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i ->
                values[i] = tanh(values[i]));
        return values;
    }
    private static double[][] tanh(double[][] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i ->
                values[i] = tanh(values[i]));
        return values;
    }
}
