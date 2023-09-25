package com.craftsentient.craftmind.errorLoss;

public class ErrorLossFunctions {
    public static void lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[] trueValues, double[] predictedValues){
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> { binaryCrossEntropy(trueValues, predictedValues); }
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { categoricalCrossEntropy(trueValues, predictedValues); }
            case CONTRASTIVE_LOSS_FUNCTION -> { contrastive(trueValues, predictedValues); }
            case COSINE_PROXIMITY_LOSS_FUNCTION -> {}
            case FOCAL_LOSS_FUNCTION -> {}
            case HINGE_LOSS_FUNCTION -> {}
            case HUBER_LOSS_FUNCTION -> {}
            case KL_DIVERGENCE_LOSS_FUNCTION -> {}
            case LOG_COSH_LOSS_FUNCTION -> {}
            case L1_LOSS_FUNCTION -> { absoluteError(trueValues, predictedValues); }
            case MAPE_LOSS_FUNCTION -> {}
            case MSE_LOSS_FUNCTION -> {}
            case MSLE_LOSS_FUNCTION -> {}
            case NLL_LOSS_FUNCTION -> {}
            case QUADRATIC_LOSS -> {}
            case RANKNET_LOSS_FUNCTION -> {}
            case SPARSE_CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> {}
            case SQUARED_HINGE_LOSS_FUNCTION -> {}
            case SSIM_LOSS_FUNCTION -> {}
            case TRIPLET_MARGIN_LOSS_FUNCTION -> {}
        }
    }
    public static void lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[][] trueValues, double[][] predictedValues){
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> { binaryCrossEntropy(trueValues, predictedValues); }
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { categoricalCrossEntropy(trueValues, predictedValues); }
            case CONTRASTIVE_LOSS_FUNCTION -> { contrastive(trueValues, predictedValues); }
            case COSINE_PROXIMITY_LOSS_FUNCTION -> {}
            case FOCAL_LOSS_FUNCTION -> {}
            case HINGE_LOSS_FUNCTION -> {}
            case HUBER_LOSS_FUNCTION -> {}
            case KL_DIVERGENCE_LOSS_FUNCTION -> {}
            case LOG_COSH_LOSS_FUNCTION -> {}
            case L1_LOSS_FUNCTION -> {}
            case MAPE_LOSS_FUNCTION -> {}
            case MSE_LOSS_FUNCTION -> {}
            case MSLE_LOSS_FUNCTION -> {}
            case NLL_LOSS_FUNCTION -> {}
            case QUADRATIC_LOSS -> {}
            case RANKNET_LOSS_FUNCTION -> {}
            case SPARSE_CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> {}
            case SQUARED_HINGE_LOSS_FUNCTION -> {}
            case SSIM_LOSS_FUNCTION -> {}
            case TRIPLET_MARGIN_LOSS_FUNCTION -> {}
        }
    }

    private static void binaryCrossEntropy(double trueValue, double predictedValue) {}
    private static void binaryCrossEntropy(double[] trueValues, double[] predictedValues) {}
    private static void binaryCrossEntropy(double[][] trueValues, double[][] predictedValues) {}


    private static void categoricalCrossEntropy(double trueValue, double predictedValue) {}
    private static void categoricalCrossEntropy(double[] trueValues, double[] predictedValues) {}
    private static void categoricalCrossEntropy(double[][] trueValues, double[][] predictedValues) {}


    private static void contrastive(double trueValue, double predictedValue) {}
    private static void contrastive(double[] trueValues, double[] predictedValues) {}
    private static void contrastive(double[][] trueValues, double[][] predictedValues) {}


    private static void cosineProximity(double trueValue, double predictedValue) {}
    private static void cosineProximity(double[] trueValues, double[] predictedValues) {}
    private static void cosineProximity(double[][] trueValues, double[][] predictedValues) {}


    private static void focal(double trueValue, double predictedValue) {}
    private static void focal(double[] trueValues, double[] predictedValues) {}
    private static void focal(double[][] trueValues, double[][] predictedValues) {}


    private static void hinge(double trueValue, double predictedValue) {}
    private static void hinge(double[] trueValues, double[] predictedValues) {}
    private static void hinge(double[][] trueValues, double[][] predictedValues) {}


    private static void huber(double trueValue, double predictedValue) {}
    private static void huber(double[] trueValues, double[] predictedValues) {}
    private static void huber(double[][] trueValues, double[][] predictedValues) {}


    private static void kullbackLeiblerDivergence(double trueValue, double predictedValue) {}
    private static void kullbackLeiblerDivergence(double[] trueValues, double[] predictedValues) {}
    private static void kullbackLeiblerDivergence(double[][] trueValues, double[][] predictedValues) {}


    private static void logCosh(double trueValue, double predictedValue) {}
    private static void logCosh(double[] trueValues, double[] predictedValues) {}
    private static void logCosh(double[][] trueValues, double[][] predictedValues) {}


    private static void absoluteError(double trueValue, double predictedValue) {}
    private static void absoluteError(double[] trueValues, double[] predictedValues) {}
    private static void absoluteError(double[][] trueValues, double[][] predictedValues) {}


    private static void meanAbsolutePercentageError(double trueValue, double predictedValue) {}
    private static void meanAbsolutePercentageError(double[] trueValues, double[] predictedValues) {}
    private static void meanAbsolutePercentageError(double[][] trueValues, double[][] predictedValues) {}


    private static void meanStandardError(double trueValue, double predictedValue) {}
    private static void meanStandardError(double[] trueValues, double[] predictedValues) {}
    private static void meanStandardError(double[][] trueValues, double[][] predictedValues) {}


    private static void meanStandardLogarithmicError(double trueValue, double predictedValue) {}
    private static void meanStandardLogarithmicError(double[] trueValues, double[] predictedValues) {}
    private static void meanStandardLogarithmicError(double[][] trueValues, double[][] predictedValues) {}


    private static void negativeLogLikelihood(double trueValue, double predictedValue) {}
    private static void negativeLogLikelihood(double[] trueValues, double[] predictedValues) {}
    private static void negativeLogLikelihood(double[][] trueValues, double[][] predictedValues) {}


    private static void QuadraticLoss(double trueValue, double predictedValue) {}
    private static void QuadraticLoss(double[] trueValues, double[] predictedValues) {}
    private static void QuadraticLoss(double[][] trueValues, double[][] predictedValues) {}


    private static void rankNet(double trueValue, double predictedValue) {}
    private static void rankNet(double[] trueValues, double[] predictedValues) {}
    private static void rankNet(double[][] trueValues, double[][] predictedValues) {}


    private static void sparseCategoricalCrossEntropy(double trueValue, double predictedValue) {}
    private static void sparseCategoricalCrossEntropy(double[] trueValues, double[] predictedValues) {}
    private static void sparseCategoricalCrossEntropy(double[][] trueValues, double[][] predictedValues) {}


    private static void squaredHinge(double trueValue, double predictedValue) {}
    private static void squaredHinge(double[] trueValues, double[] predictedValues) {}
    private static void squaredHinge(double[][] trueValues, double[][] predictedValues) {}


    private static void structuralSimilarityIndex(double trueValue, double predictedValue) {}
    private static void structuralSimilarityIndex(double[] trueValues, double[] predictedValues) {}
    private static void structuralSimilarityIndex(double[][] trueValues, double[][] predictedValues) {}


    private static void tripletMargin(double trueValue, double predictedValue) {}
    private static void tripletMargin(double[] trueValues, double[] predictedValues) {}
    private static void tripletMargin(double[][] trueValues, double[][] predictedValues) {}
}
