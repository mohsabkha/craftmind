package com.craftsentient.craftmind.errorLoss;

import static com.craftsentient.craftmind.errorLoss.ErrorLossFunctionsImpl.*;


public class ErrorLossFunctions {
    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex, int selectedOutputIndex, double[] outputs) {
        return determineLossFunctionFromIndices(lossFunction, trueValueIndex, selectedOutputIndex, outputs);
    }
    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int[] hotOneVec, int selectedOutputIndex, double[] outputs) {
        int trueValueIndex = 0;

        return determineLossFunctionFromIndices(lossFunction, trueValueIndex, selectedOutputIndex, outputs);
    }

    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndices[], int selectedOutputIndex[], double[][] outputs) {
        return determineLossFunctionFromIndices(lossFunction, trueValueIndices, selectedOutputIndex, outputs);
    }
    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int hotOneVectors[][], int selectedOutputIndex[], double[][] outputs) {
        int trueValueIndices[] = new int[hotOneVectors.length];
        for(int i = 0; i < hotOneVectors.length; i++) {
            for(int j = 0; j < hotOneVectors[i].length; j++){
                if(hotOneVectors[i][j] != 0){
                    trueValueIndices[i] = j;
                    break;
                }
            }

        }
        return determineLossFunctionFromIndices(lossFunction, trueValueIndices, selectedOutputIndex, outputs);
    }

    // true value index and hot one vector
    private static double determineLossFunctionFromIndices(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex, int selectedOutputIndex, double[] outputs) {
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> { return binaryCrossEntropy(trueValueIndex, selectedOutputIndex, outputs); }
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return categoricalCrossEntropy(trueValueIndex, outputs); }
            case FOCAL_LOSS_FUNCTION -> { return focal(trueValueIndex, outputs); }
            case HINGE_LOSS_FUNCTION -> { return hinge(trueValueIndex, outputs); }
            case HUBER_LOSS_FUNCTION -> { return huber(trueValueIndex, outputs); }
            case LOG_COSH_LOSS_FUNCTION -> { return logCosh(trueValueIndex, outputs); }
            case MSLE_LOSS_FUNCTION -> {  return meanStandardLogarithmicError(trueValueIndex, selectedOutputIndex, outputs); }
            case NLL_LOSS_FUNCTION -> { return negativeLogLikelihood(trueValueIndex, outputs); }
            case SQUARED_HINGE_LOSS_FUNCTION -> { return squaredHinge(trueValueIndex, outputs); }
            default -> throw new RuntimeException("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    private static double[] determineLossFunctionFromIndices(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex, int selectedOutputIndex, double[][] outputs) {
        return new double[0];
    }
    private static double[][] determineLossFunctionFromIndices(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex, int selectedOutputIndex, double[][][] outputs) {
        return new double[0][];
    }
    private static double[][][] determineLossFunctionFromIndices(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex, int selectedOutputIndex, double[][][][] outputs) {
        return new double[0][][];
    }

    // true value indices and hot one vectors (batch)
    private static double determineLossFunctionFromIndices(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex[], int selectedOutputIndex[], double[][] outputs) {
        return 0.0;
    }
    private static double[] determineLossFunctionFromIndices(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex[], int selectedOutputIndex[], double[][][] outputs) {
        return new double[0];
    }
    private static double[][] determineLossFunctionFromIndices(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex[], int selectedOutputIndex[], double[][][][]outputs) {
        return new double[0][];
    }

    // true value indices and hot one vectors
    private static double[][] determineLossFunctionFromIndices(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex[][], int selectedOutputIndex[][], double[][][] outputs) {
        return new double[0][];
    }
    private static double[][][] determineLossFunctionFromIndices(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex[][], int selectedOutputIndex[][], double[][][][] outputs) {
        return new double[0][][];
    }

}
