package com.craftsentient.craftmind.derivitives.activationDerivatives;

import com.craftsentient.craftmind.activation.DEFAULT_ACTIVATION_FUNCTIONS;

import static com.craftsentient.craftmind.derivitives.activationDerivatives.ActivationImpl.*;

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
                return parametricRelu(value, alpha);
            }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }

    public static double[] derivative(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double[] values) {
        switch (activationFunction) {
            case BENT_IDENTITY_ACTIVATION_FUNCTION -> {
                return bentIdentity(values);
            }
            case GAUSSIAN_ACTIVATION_FUNCTION -> {
                return gaussian(values);
            }
            case HARD_SIGMOID_ACTIVATION_FUNCTION -> {
                return hardSigmoid(values);
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
            case RELU_6_ACTIVATION_FUNCTION -> {
                return rectifiedLinearUnit6(values);
            }
            case SELU_ACTIVATION_FUNCTION -> {
                return scaledExponentialLinear(values);
            }
            case SIGMOID_ACTIVATION_FUNCTION -> {
                return sigmoid(values);
            }
            case SOFTMAX_ACTIVATION_FUNCTION -> {
                return cappedSoftmax(values);
            }
            case SOFTPLUS_ACTIVATION_FUNCTION -> {
                return softplus(values);
            }
            case SOFTSIGN_ACTIVATION_FUNCTION -> {
                return softsign(values);
            }
            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(values);
            }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }

    public static double[] derivative(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double values[], double alphas[], double betas[]) {
        switch (activationFunction) {
            case SWISH_ACTIVATION_FUNCTION -> {
                return swish(values, betas);
            }
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> {
                return exponentialElu(values, alphas, betas);
            }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> {
                return parametricRelu(values, alphas);
            }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }

    public static double[][] derivative(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double[][] values) {
        switch (activationFunction) {
            case BENT_IDENTITY_ACTIVATION_FUNCTION -> {
                return bentIdentity(values);
            }
            case GAUSSIAN_ACTIVATION_FUNCTION -> {
                return gaussian(values);
            }
            case HARD_SIGMOID_ACTIVATION_FUNCTION -> {
                return hardSigmoid(values);
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
            case RELU_6_ACTIVATION_FUNCTION -> {
                return rectifiedLinearUnit6(values);
            }
            case SELU_ACTIVATION_FUNCTION -> {
                return scaledExponentialLinear(values);
            }
            case SIGMOID_ACTIVATION_FUNCTION -> {
                return sigmoid(values);
            }
            case SOFTMAX_ACTIVATION_FUNCTION -> {
                return cappedSoftmax(values);
            }
            case SOFTPLUS_ACTIVATION_FUNCTION -> {
                return softplus(values);
            }
            case SOFTSIGN_ACTIVATION_FUNCTION -> {
                return softsign(values);
            }
            case TANH_ACTIVATION_FUNCTION -> {
                return tanh(values);
            }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }

    public static double[][] derivative(DEFAULT_ACTIVATION_FUNCTIONS activationFunction, double values[][], double alphas[][], double betas[][]) {
        switch (activationFunction) {
            case SWISH_ACTIVATION_FUNCTION -> {
                return swish(values, betas);
            }
            case EXPONENTIAL_ELU_ACTIVATION_FUNCTION -> {
                return exponentialElu(values, alphas, betas);
            }
            case PARAMETRIC_RELU_ACTIVATION_FUNCTION -> {
                return parametricRelu(values, alphas);
            }
            default -> throw new RuntimeException("Incorrect Activation Function Name Entered: " + activationFunction.name());
        }
    }
}

