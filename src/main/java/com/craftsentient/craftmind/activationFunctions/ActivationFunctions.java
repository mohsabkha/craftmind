package com.craftsentient.craftmind.activationFunctions;

import java.util.Arrays;
import java.util.stream.IntStream;

public class ActivationFunctions {
    public static double activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double value) throws Exception {
        switch (activationFunction) {

            case GAUSSIAN_ACTIVATION_FUNCTION -> {
                return gaussian(value);
            }

            case LINEAR_ACTIVATION_FUNCTION -> {
                return linear(value);
            }
            case MISH_ACTIVATION_FUNCTION -> {
                return mish(value);
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

            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(value);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }
    public static double activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double value, double alpha, double beta) throws Exception {
        switch (activationFunction) {
            case SWISH_ACTIVATION_FUNCTION -> {
                return swish(value, beta);
            }
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> {
                return exponentialElu(value, alpha, beta);
            }
            case LEAKY_RELU_ACTIVATION_FUNCTION -> {
                return leakyRelu(value, alpha);
            }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> {
                return parametricRelu(value,alpha);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }

    public static double[] activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double[] values) throws Exception {
        switch (activationFunction) {

            case GAUSSIAN_ACTIVATION_FUNCTION -> {
                return gaussian(values);
            }
            case LINEAR_ACTIVATION_FUNCTION -> {
                return linear(values);
            }
            case MISH_ACTIVATION_FUNCTION -> {
                return mish(values);
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
            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(values);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }
    public static double[] activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double values[], double alphas[], double betas[]) throws Exception {
        switch (activationFunction) {
            case SWISH_ACTIVATION_FUNCTION -> {
                return swish(values, betas);
            }
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> {
                return exponentialElu(values, alphas, betas);
            }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> {
                return parametricRelu(values,alphas);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }

    public static double[][] activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double[][] values) throws Exception {
        switch (activationFunction) {
            case GAUSSIAN_ACTIVATION_FUNCTION -> {
                return gaussian(values);
            }
            case LINEAR_ACTIVATION_FUNCTION -> {
                return linear(values);
            }
            case MISH_ACTIVATION_FUNCTION -> {
                return mish(values);
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
            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(values);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }
    public static double[][] activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double values[][], double alphas[][], double betas[][]) throws Exception {
        switch (activationFunction) {
            case SWISH_ACTIVATION_FUNCTION -> {
                return swish(values, betas);
            }
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> {
                return exponentialElu(values, alphas, betas);
            }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> {
                return parametricRelu(values,alphas);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }



    // EXPONENTIAL_ELU
    private static double exponentialElu(double value,double alpha, double beta){
        if (value < 0) {
            return alpha * (Math.exp(value) - 1);
        } else {
            return beta * Math.exp(value);
        }
    }
    private static double[] exponentialElu(double values[],double alphas[], double betas[]){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> {
            values[i] = exponentialElu(values[i], alphas[i], betas[i]);
        });
        return values;
    }
    private static double[][] exponentialElu(double values[][],double alphas[][], double betas[][]){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> {
            values[i] = exponentialElu(values[i], alphas[i], betas[i]);
        });
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
    private static double leakyRelu(double value, double alpha){
        if (value > 0) {
            return value;
        } else {
            return alpha * value;
        }
    }
    private static double[] leakyRelu(double[] values, double alpha){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = leakyRelu(values[i], alpha));
        return values;
    }
    private static double[][] leakyRelu(double[][] values, double alpha){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = leakyRelu(values[i], alpha));
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
        return value * Math.tanh(Math.log1p(Math.exp(value)));
    }
    private static double[] mish(double[] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = mish(values[i]));
        return values;
    }
    private static double[][] mish(double[][] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = mish(values[i]));
        return values;
    }


    // PARAMETRIC_RELU
    private static double parametricRelu(double value, double alpha){
        if (value > 0) {
            return value;
        } else {
            return alpha * value;
        }
    }
    private static double[] parametricRelu(double[] values, double[] alphas){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = parametricRelu(values[i], alphas[i]));
        return values;
    }
    private static double[][] parametricRelu(double[][] values, double[][] alphas){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = parametricRelu(values[i], alphas[i]));
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
        return Math.log1p(Math.exp(value));
    }
    private static double[] softplus(double[] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = softplus(values[i]));
        return values;
    }
    private static double[][] softplus(double[][] values){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = softplus(values[i]));
        return values;
    }


    // SWISH
    private static double swish(double value, double beta){
        return value * sigmoid(value);
    }
    private static double[] swish(double[] values, double[] beta) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = beta[i] * sigmoid(values[i]));
        return values;
    }
    private static double[][] swish(double[][] values, double[][] beta){
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = swish(values[i], beta[i]));
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
