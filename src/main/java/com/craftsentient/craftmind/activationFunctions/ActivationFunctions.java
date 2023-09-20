package com.craftsentient.craftmind.activationFunctions;

import java.util.Arrays;

public class ActivationFunctions {
    public static double activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double value) throws Exception {
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
            case SOFTMAX_ACTIVATION_FUNCTION -> {
                return softmax(value);
            }
            case SOFTPLUS_ACTIVATION_FUNCTION -> {
                return softplus(value);
            }
            case SWISH_ACTIVATION_FUNCTION -> {
                return swish(value);
            }
            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(value);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }

    public static double[] activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double[] values) throws Exception {
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
                return swish(values);
            }
            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(values);
            }
            default -> throw new Exception("Incorrect Activation Function Name Entered: " + activationFunction.name());

        }
    }

    public static double[][] activationFunction(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double[][] values) throws Exception {
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
                return swish(values);
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
        return value;
    }
    private static double[] gaussian(double[] values){
        return values;
    }
    private static double[][] gaussian(double[][] values){
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
        return  value;
    }
    private static double[]sigmoid(double[] values){
        return  values;
    }
    private static double[][] sigmoid(double[][] values){
        return  values;
    }


    // SOFTMAX
    private static double softmax(double value){
        return  value;
    }
    private static double[] softmax(double[] values){
        return  values;
    }
    private static double[][] softmax(double[][] values){
        return  values;
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
    private static double swish(double value){
        return value;
    }
    private static double[] swish(double[] values){
        return values;
    }
    private static double[][] swish(double[][] values){
        return values;
    }


    // TANH
    private static double tanh(double value){
        return value;
    }
    private static double[] tanh(double[] values){
        return values;
    }
    private static double[][] tanh(double[][] values){
        return values;
    }
}
