package com.craftsentient.craftmind.derivitives;

import com.craftsentient.craftmind.activation.ActivationFunctions;
import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;

import java.util.stream.IntStream;

public class Activation {
    public static double derivative(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double value) {
        switch (activationFunction) {
            case BENT_IDENTITY_ACTIVATION_FUNCTION -> {
                return bentIdentity(value);
            }
            case GAUSSIAN_ACTIVATION_FUNCTION -> {
                return gaussian(value);
            }
            case HARD_SIGMOID_ACTIVATION_FUNCTION -> {
                return hardSigmoid(value);
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
            case RELU_6_ACTIVATION_FUNCTION -> {
                return rectifiedLinearUnit6(value);
            }
            case SELU_ACTIVATION_FUNCTION -> {
                return scaledExponentialLinear(value);
            }
            case SIGMOID_ACTIVATION_FUNCTION -> {
                return sigmoid(value);
            }
            case SOFTPLUS_ACTIVATION_FUNCTION -> {
                return softplus(value);
            }
            case SOFTSIGN_ACTIVATION_FUNCTION -> {
                return softsign(value);
            }
            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(value);
            }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }
    public static double derivative(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double value, double alpha, double beta) {
        switch (activationFunction) {
            case SWISH_ACTIVATION_FUNCTION -> { return swish(value, beta); }
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> { return exponentialElu(value, alpha, beta); }
            case LEAKY_RELU_ACTIVATION_FUNCTION -> { return leakyRelu(value, alpha); }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> { return parametricRelu(value,alpha); }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }
    public static double[] derivative(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double[] values) {
        switch (activationFunction) {
            case BENT_IDENTITY_ACTIVATION_FUNCTION -> { return bentIdentity(values); }
            case GAUSSIAN_ACTIVATION_FUNCTION -> { return gaussian(values); }
            case HARD_SIGMOID_ACTIVATION_FUNCTION -> { return hardSigmoid(values); }
            case LINEAR_ACTIVATION_FUNCTION -> { return linear(values); }
            case MISH_ACTIVATION_FUNCTION -> { return mish(values); }
            case RELU_ACTIVATION_FUNCTION -> { return rectifiedLinearUnit(values); }
            case RELU_6_ACTIVATION_FUNCTION -> { return rectifiedLinearUnit6(values); }
            case SELU_ACTIVATION_FUNCTION -> { return scaledExponentialLinear(values); }
            case SIGMOID_ACTIVATION_FUNCTION -> { return sigmoid(values); }
            case SOFTMAX_ACTIVATION_FUNCTION -> { return cappedSoftmax(values); }
            case SOFTPLUS_ACTIVATION_FUNCTION -> { return softplus(values); }
            case SOFTSIGN_ACTIVATION_FUNCTION -> { return softsign(values); }
            case TANH_ACTIVATION_FUNCTION -> { return tanh(values); }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }
    public static double[] derivative(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double values[], double alphas[], double betas[]) {
        switch (activationFunction) {
            case SWISH_ACTIVATION_FUNCTION -> { return swish(values, betas); }
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> { return exponentialElu(values, alphas, betas); }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> { return parametricRelu(values,alphas); }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }
    public static double[][] derivative(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double[][] values) {
        switch (activationFunction) {
            case BENT_IDENTITY_ACTIVATION_FUNCTION -> { return bentIdentity(values); }
            case GAUSSIAN_ACTIVATION_FUNCTION -> { return gaussian(values); }
            case HARD_SIGMOID_ACTIVATION_FUNCTION -> { return hardSigmoid(values); }
            case LINEAR_ACTIVATION_FUNCTION -> { return linear(values); }
            case MISH_ACTIVATION_FUNCTION -> { return mish(values); }
            case RELU_ACTIVATION_FUNCTION -> { return rectifiedLinearUnit(values); }
            case RELU_6_ACTIVATION_FUNCTION -> { return rectifiedLinearUnit6(values); }
            case SELU_ACTIVATION_FUNCTION -> { return scaledExponentialLinear(values); }
            case SIGMOID_ACTIVATION_FUNCTION -> { return sigmoid(values); }
            case SOFTMAX_ACTIVATION_FUNCTION -> { return cappedSoftmax(values); }
            case SOFTPLUS_ACTIVATION_FUNCTION -> { return softplus(values); }
            case SOFTSIGN_ACTIVATION_FUNCTION -> { return softsign(values); }
            case TANH_ACTIVATION_FUNCTION -> { return tanh(values); }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }
    public static double[][] derivative(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double values[][], double alphas[][], double betas[][]) {
        switch (activationFunction) {
            case SWISH_ACTIVATION_FUNCTION -> { return swish(values, betas); }
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> { return exponentialElu(values, alphas, betas); }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> { return parametricRelu(values,alphas); }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }




    //BENT IDENTITY
    public static double bentIdentity(double value) {
        return (value/(2* Math.sqrt((value * value) + 1))) + 1;

    }
    public static double[] bentIdentity(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = bentIdentity(values[i]));
        return values;
    }
    public static double[][] bentIdentity(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = bentIdentity(values[i]));
        return values;
    }


    // EXPONENTIAL_LU
    public static double exponentialElu(double value,double alpha, double beta) {
        if (value <= 0) return alpha * Math.exp(value);
        else return beta * Math.exp(value);
    }
    public static double[] exponentialElu(double values[],double alphas[], double betas[]) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = exponentialElu(values[i], alphas[i], betas[i]));
        return values;
    }
    public static double[][] exponentialElu(double values[][],double alphas[][], double betas[][]) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = exponentialElu(values[i], alphas[i], betas[i]));
        return values;
    }


    // GAUSSIAN_ACTIVATION
    public static double gaussian(double value) {
        return (-2 * value) * Math.exp(-(value*value));
    }
    public static double[] gaussian(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = gaussian(values[i]));
        return values;
    }
    public static double[][] gaussian(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = gaussian(values[i]));
        return values;
    }


    // HARD SIGMOID
    public static double hardSigmoid(double value) {
        if (value < -2.5) return 0.0;
        else if (value > 2.5) return 0.0;
        else return 0.2;
    }
    public static double[] hardSigmoid(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = hardSigmoid(values[i]));
        return values;
    }
    public static double[][] hardSigmoid(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = hardSigmoid(values[i]));
        return values;
    }


    // LEAKY_RELU
    public static double leakyRelu(double value, double alpha) {
        if (value > 0) return 1;
        else return alpha;
    }
    public static double[] leakyRelu(double[] values, double alpha) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = leakyRelu(values[i], alpha));
        return values;
    }
    public static double[][] leakyRelu(double[][] values, double alpha) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = leakyRelu(values[i], alpha));
        return values;
    }


    // LINEAR_ACTIVATION
    public static double linear(double value) {
        return 1;
    }
    public static double[] linear(double[] values) {
        IntStream.range(0, values.length).forEach(i -> values[i] = 1);
        return values;
    }
    public static double[][] linear(double[][] values) {
        IntStream.range(0, values.length).parallel().forEach(i -> values[i] = linear(values[i]));
        return values;
    }


    // MISH_ACTIVATION
    public static double mish(double value) {
        return
                ActivationFunctions.tanh(ActivationFunctions.softplus(value)) +
                        (value * (1 - (ActivationFunctions.tanh(ActivationFunctions.softplus(value)) * ActivationFunctions.tanh(ActivationFunctions.softplus(value)))) * ActivationFunctions.sigmoid(value));
    }
    public static double[] mish(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> {
            try {
                values[i] = mish(values[i]);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        return values;
    }
    public static double[][] mish(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = mish(values[i]));
        return values;
    }


    // PARAMETRIC_RELU
    public static double parametricRelu(double value, double alpha) {
        if (value > 0){ return 1; }
        else return alpha;
    }
    public static double[] parametricRelu(double[] values, double[] alphas) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = parametricRelu(values[i], alphas[i]));
        return values;
    }
    public static double[][] parametricRelu(double[][] values, double[][] alphas) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = parametricRelu(values[i], alphas[i]));
        return values;
    }


    // RELU_ACTIVATION
    public static double rectifiedLinearUnit(double value) {
        if(value <= 0){ return 0; }
        return 1;
    }
    public static double[] rectifiedLinearUnit(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = rectifiedLinearUnit(values[i]));
        return values;
    }
    public static double[][] rectifiedLinearUnit(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = rectifiedLinearUnit(values[i]));
        return values;
    }


    // RELU6
    public static double rectifiedLinearUnit6(double value) {
        if(value < 0 || value > 6){ return 0; }
        return 1;
    }
    public static double[] rectifiedLinearUnit6(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = rectifiedLinearUnit6(values[i]));
        return values;
    }
    public static double[][] rectifiedLinearUnit6(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = rectifiedLinearUnit6(values[i]));
        return values;
    }


    // SELU
    public static double scaledExponentialLinear(double value) {
        double LAMBDA = 1.0507;
        double ALPHA = 1.67326;
        if (value > 0) return LAMBDA;
        else return LAMBDA * (ALPHA + ActivationFunctions.activationFunction(DEFAULT_ACTIVATION_FUNCTIONS.SELU_ACTIVATION_FUNCTION, value));
    }
    public static double[] scaledExponentialLinear(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> {
            try {
                values[i] = scaledExponentialLinear(values[i]);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        return values;
    }
    public static double[][] scaledExponentialLinear(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = scaledExponentialLinear(values[i]));
        return values;
    }


    // SIGMOID_ACTIVATION
    public static double sigmoid(double value) {
        return ActivationFunctions.sigmoid(value) *
                (1 - ActivationFunctions.sigmoid(value));
    }
    public static double[] sigmoid(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> {
            try {
                values[i] = sigmoid(values[i]);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        return values;
    }
    public static double[][] sigmoid(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = sigmoid(values[i]));
        return values;
    }


    // TODO: Implement Derivative
    // SOFTMAX
    public static double[] softmax(double[] values) {
        double sum = 0.0;
        for (double value : values) { sum += Math.exp(value); }
        for (int i = 0; i < values.length; i++) { values[i] = Math.exp(values[i]) / sum; }
        return values;
    }
    public static double[][] softmax(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = softmax(values[i]));
        return values;
    }

    // TODO: implement finding the largest value in the layer itself as neurons are added
    // TODO: Implement Derivative
    // CAPPED SOFTMAX
    public static double[] cappedSoftmax(double[] values) {
        double sum = 0.0;
        double largest = 0;
        for (double value : values) { if (value > largest) largest = value; }
        // subtract largest from max values to verify no overflow
        for (double value : values) { sum += Math.exp(value-largest); }
        for (int i = 0; i < values.length; i++) { values[i] = Math.exp(values[i]-largest) / sum; }
        return values;
    }
    public static double[][] cappedSoftmax(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = softmax(values[i]));
        return values;
    }


    // SOFTPLUS
    public static double softplus(double value) {
        return ActivationFunctions.sigmoid(value);
    }
    public static double[] softplus(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> {
            try {
                values[i] = softplus(values[i]);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        return values;
    }
    public static double[][] softplus(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = softplus(values[i]));
        return values;
    }


    // SOFTSIGN
    public static double softsign(double value) {
        return (1 / ( (1 + Math.abs(value)*(1 + Math.abs(value))) ));
    }
    public static double[] softsign(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = softsign(values[i]));
        return values;
    }
    public static double[][] softsign(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = softsign(values[i]));
        return values;
    }


    // SWISH
    public static double swish(double value, double beta) {
        return (1 + value - (value * ActivationFunctions.sigmoid(value))) * ActivationFunctions.sigmoid(value);
    }
    public static double[] swish(double[] values, double[] beta) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> {
            try {
                values[i] = beta[i] * sigmoid(values[i]);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        return values;
    }
    public static double[][] swish(double[][] values, double[][] beta) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = swish(values[i], beta[i]));
        return values;
    }


    // TANH
    public static double tanh(double value) {
        return (1 - (tanh(value) * tanh(value)));
    }
    public static double[] tanh(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = tanh(values[i]));
        return values;
    }
    public static double[][] tanh(double[][] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = tanh(values[i]));
        return values;
    }
}
