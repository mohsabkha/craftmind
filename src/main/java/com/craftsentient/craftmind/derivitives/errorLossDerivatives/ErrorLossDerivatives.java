package com.craftsentient.craftmind.derivitives.errorLossDerivatives;

import com.craftsentient.craftmind.errorLoss.DEFAULT_LOSSES;

import static com.craftsentient.craftmind.derivitives.errorLossDerivatives.ErrorLossDerivativesImpl.*;

public class ErrorLossDerivatives {
    public static double[] derivative(DEFAULT_LOSSES lossFunction, int trueValueIndex, int selectedOutputIndex, double[] outputs){
        switch (lossFunction) {
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> { return binaryCrossEntropy(trueValueIndex, outputs); }
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return categoricalCrossEntropy(trueValueIndex, outputs); }
            case FOCAL_LOSS_FUNCTION -> { return focal(trueValueIndex, outputs); }
            case HINGE_LOSS_FUNCTION -> { return hinge(trueValueIndex, outputs); }
            case HUBER_LOSS_FUNCTION -> { return huber(trueValueIndex, outputs); }
            case LOG_COSH_LOSS_FUNCTION -> { return logCosh(trueValueIndex, outputs); }
            case MSLE_LOSS_FUNCTION -> {  return meanStandardLogarithmicError(trueValueIndex, outputs); }
            case NLL_LOSS_FUNCTION -> { return negativeLogLikelihood(trueValueIndex, outputs); }
            case SQUARED_HINGE_LOSS_FUNCTION -> { return squaredHinge(trueValueIndex, outputs); }
            default -> throw new RuntimeException("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double[] derivative(DEFAULT_LOSSES lossFunction, int[] hotOneVector, int selectedOutputIndex, double[] outputs){
        int trueValueIndex = 0;
        for(int i = 0; i < hotOneVector.length; i++) {
            if(hotOneVector[i] != 0){
                trueValueIndex = i;
                break;
            }
        }
        return derivative(lossFunction, trueValueIndex, selectedOutputIndex, outputs);
    }
}
