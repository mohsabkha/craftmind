package com.craftsentient.craftmind.derivitives.errorLossDerivatives;

import static com.craftsentient.craftmind.layers.DenseLayers.*;
import static com.craftsentient.craftmind.layers.DenseLayers.DELTA;

public class ErrorLossImpl {
    public static double[] binaryCrossEntropy(int trueValueIndex, double[] outputs){
        double[] derivatives = new double[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            if (i == trueValueIndex) {
                // For the correct class
                derivatives[i] = -1 / outputs[i];
            } else {
                // For all other classes
                derivatives[i] = 1 / (1 - outputs[i]);
            }
        }
        return derivatives;
    }

    public static double[] categoricalCrossEntropy(int trueValueIndex, double[] outputs){
        double[] derivatives = new double[outputs.length];

        for (int i = 0; i < outputs.length; i++) {
            if (i == trueValueIndex) { derivatives[i] = -(1 - outputs[i]); }
            else { derivatives[i] = outputs[i];}
        }
        return derivatives;
    }

    public static double[] focal(int trueValueIndex, double[] outputs) {
        double[] gradients = new double[outputs.length];
        double pt = outputs[trueValueIndex];

        for (int i = 0; i < outputs.length; i++) {
            double modulatingFactor = Math.pow(1 - pt, GAMMA);
            if (i == trueValueIndex) {
                gradients[i] = modulatingFactor * (GAMMA * pt * Math.log(pt) + pt - 1);
            } else {
                gradients[i] = modulatingFactor * outputs[i] * (GAMMA * Math.log(pt) + 1);
            }
        }
        return gradients;
    }

    public static double[] hinge(int trueValueIndex, double[] outputs) {
        double[] subgradients = new double[outputs.length];

        // Find the score of the true class and the maximum score among the incorrect classes
        double trueScore = outputs[trueValueIndex];
        double maxIncorrectScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < outputs.length; i++) {
            if (i != trueValueIndex) {
                maxIncorrectScore = Math.max(maxIncorrectScore, outputs[i]);
            }
        }
        // Check if there is any loss
        if (maxIncorrectScore + MARGIN > trueScore) {
            // If there's a loss, set the subgradient for the true class
            subgradients[trueValueIndex] = -1;
            // Set the subgradient for any incorrect class that is within the margin
            for (int i = 0; i < outputs.length; i++) {
                if (i != trueValueIndex && outputs[i] + MARGIN > trueScore) {
                    subgradients[i] = 1;
                }
            }
        }
        // If there's no loss, all subgradients are 0, which is the default value in the array
        return subgradients;
    }

    public static double[] huber(int trueValueIndex, double[] outputs) {
        double trueValue = outputs[trueValueIndex];
        double[] gradients = new double[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            double error = outputs[i] - (i == trueValueIndex ? trueValue : 0);
            // Check whether the error is within the delta boundary
            if (Math.abs(error) <= DELTA) {
                // Quadratic loss zone
                gradients[i] = error;
            } else {
                // Linear loss zone
                gradients[i] = DELTA * Math.signum(error);
            }
        }
        return gradients;
    }

    public static double[] logCosh(double trueValueIndex, double[] outputs) {
        double[] gradients = new double[outputs.length];
        // Assuming the true value at trueIndex is 1 and 0 for all other classes
        for (int i = 0; i < outputs.length; i++) {
            if (i == trueValueIndex) {
                // For the true class
                gradients[i] = Math.tanh(outputs[i] - 1);
            } else {
                // For all other classes
                gradients[i] = Math.tanh(outputs[i]);
            }
        }
        return gradients;
    }

    public static double[] meanStandardLogarithmicError(int trueValueIndex, double[] outputs) {
        double[] gradients = new double[outputs.length];

        for (int i = 0; i < outputs.length; i++) {
            // The derivative of MSLE where the true target is 1 for the true index and 0 for all others.
            // Add 1 to both true and predicted to avoid taking the log of 0.
            if (i == trueValueIndex) {
                // For the true class, avoid log(0) by ensuring the output is positive
                double output = Math.max(outputs[i], 1e-15);  // Add epsilon to avoid log(0)
                gradients[i] = - (1 / output) * (Math.log(1 + 1) - Math.log(1 + output));
            } else {
                // For non-true classes, if outputs are 0, the gradient is 0 as well because log(1) - log(1) = 0
                if (outputs[i] > 0) {
                    double output = Math.max(outputs[i], 1e-15);  // Add epsilon to avoid log(0)
                    gradients[i] = (1 / output) * (Math.log(1) - Math.log(1 + output));
                } else {
                    gradients[i] = 0; // If the output is exactly 0, then the gradient is 0.
                }
            }
        }

        return gradients;
    }

    public static double[] negativeLogLikelihood(int trueClass, double[] predictedValues) {
        double[] gradient = new double[predictedValues.length];
        gradient[trueClass] = -1.0 / predictedValues[trueClass];
        return gradient;
    }

    public static double[] squaredHinge(int trueValueIndex, double[] outputs) {
        double[] derivatives = new double[outputs.length];
        double sumPositiveMargins = 0.0;
        // Calculate the sum of the positive margins for the true class
        for (int j = 0; j < outputs.length; j++) {
            if (j != trueValueIndex) { sumPositiveMargins += Math.max(0, 1 - (outputs[trueValueIndex] - outputs[j])); }
        }
        // Calculate the derivative for each class
        for (int k = 0; k < outputs.length; k++) {
            if (k == trueValueIndex) { derivatives[k] = -2 * sumPositiveMargins; }
            else {
                double margin = 1 - (outputs[trueValueIndex] - outputs[k]);
                if (margin > 0) { derivatives[k] = 2 * margin; }
                else { derivatives[k] = 0; }
            }
        }
        return derivatives;
    }
}
