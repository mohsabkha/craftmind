package com.craftsentient.craftmind.errorLoss;

import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

public class ErrorLossFunctions {

    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[] trueValues, double[] predictedValues) throws Exception {
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> { return binaryCrossEntropy(trueValues, predictedValues); }
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return categoricalCrossEntropy(trueValues, predictedValues); }
            // case CONTRASTIVE_LOSS_FUNCTION -> { return contrastive(trueValues, predictedValues); }
             case COSINE_PROXIMITY_LOSS_FUNCTION -> { return cosineProximity(trueValues, predictedValues); }
            case FOCAL_LOSS_FUNCTION -> { return focal(trueValues, predictedValues); }
            case HINGE_LOSS_FUNCTION -> { return hinge(trueValues, predictedValues); }
            case HUBER_LOSS_FUNCTION -> { return huber(trueValues, predictedValues); }
            case KL_DIVERGENCE_LOSS_FUNCTION -> { return kullbackLeiblerDivergence(trueValues, predictedValues); }
            case LOG_COSH_LOSS_FUNCTION -> { return logCosh(trueValues, predictedValues); }
            case L1_LOSS_FUNCTION -> { return absoluteError(trueValues, predictedValues); }
            case MAPE_LOSS_FUNCTION -> { return meanAbsolutePercentageError(trueValues, predictedValues); }
            case MSE_LOSS_FUNCTION -> { return meanStandardError(trueValues, predictedValues); }
            case MSLE_LOSS_FUNCTION -> {  return meanStandardLogarithmicError(trueValues, predictedValues); }
            case NLL_LOSS_FUNCTION -> { return negativeLogLikelihood(trueValues, predictedValues); }
            case QUADRATIC_LOSS -> { return quadraticLoss(trueValues, predictedValues); }
            case RANKNET_LOSS_FUNCTION -> { return rankNet(trueValues, predictedValues); }
            case SPARSE_CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return sparseCategoricalCrossEntropy(trueValues, predictedValues); }
            case SQUARED_HINGE_LOSS_FUNCTION -> { return squaredHinge(trueValues, predictedValues); }
            case SSIM_LOSS_FUNCTION -> { return structuralSimilarityIndex(trueValues, predictedValues); }
            case TRIPLET_MARGIN_LOSS_FUNCTION -> { return tripletMargin(trueValues, predictedValues); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }

    public static double[] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[][] trueValues, double[][] predictedValues) throws Exception {
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> { return binaryCrossEntropy(trueValues, predictedValues); }
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return categoricalCrossEntropy(trueValues, predictedValues); }
            case COSINE_PROXIMITY_LOSS_FUNCTION -> { return cosineProximity(trueValues, predictedValues); }
            case FOCAL_LOSS_FUNCTION -> { return focal(trueValues, predictedValues); }
            case HINGE_LOSS_FUNCTION -> { return hinge(trueValues, predictedValues); }
            case HUBER_LOSS_FUNCTION -> { return huber(trueValues, predictedValues); }
            case KL_DIVERGENCE_LOSS_FUNCTION -> { return kullbackLeiblerDivergence(trueValues, predictedValues); }
            case LOG_COSH_LOSS_FUNCTION -> { return logCosh(trueValues, predictedValues); }
            case L1_LOSS_FUNCTION -> { return absoluteError(trueValues, predictedValues); }
            case MAPE_LOSS_FUNCTION -> { return meanAbsolutePercentageError(trueValues, predictedValues); }
            case MSE_LOSS_FUNCTION -> { return meanStandardError(trueValues, predictedValues); }
            case MSLE_LOSS_FUNCTION -> {  return meanStandardLogarithmicError(trueValues, predictedValues); }
            case NLL_LOSS_FUNCTION -> { return negativeLogLikelihood(trueValues, predictedValues); }
            case QUADRATIC_LOSS -> { return quadraticLoss(trueValues, predictedValues); }
            case RANKNET_LOSS_FUNCTION -> { return rankNet(trueValues, predictedValues); }
            case SPARSE_CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return sparseCategoricalCrossEntropy(trueValues, predictedValues); }
            case SQUARED_HINGE_LOSS_FUNCTION -> { return squaredHinge(trueValues, predictedValues); }
            case SSIM_LOSS_FUNCTION -> { return structuralSimilarityIndex(trueValues, predictedValues); }
            case TRIPLET_MARGIN_LOSS_FUNCTION -> { return tripletMargin(trueValues, predictedValues); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }

    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[][] trueValues, double[][] predictedValues, int y, double margin){
        switch (lossFunction){
             case CONTRASTIVE_LOSS_FUNCTION -> { return contrastive(trueValues, predictedValues); }
        }
    }

    public static double[] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[][] trueValues, double[][] predictedValues, int y, double margin){

    }


    private static double binaryCrossEntropy(double[] trueValues, double[] predictedValues) {
        AtomicReference<Double> loss = new AtomicReference<>((double) 0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss.updateAndGet(v -> new Double( (double)((v + -trueValues[i] * Math.log(predictedValues[i]) - (1 - trueValues[i]) * Math.log(1 - predictedValues[i]))))) );
        return loss.get() / trueValues.length;
    }
    private static double[] binaryCrossEntropy(double[][] trueValues, double[][] predictedValues) {
        double [] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = binaryCrossEntropy(trueValues[i], predictedValues[i]));
        return loss;
    }


    private static double categoricalCrossEntropy(double[] trueValues, double[] predictedValues) {
        AtomicReference<Double> loss = new AtomicReference<>((double) 0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss.updateAndGet(v -> new Double( (double)(v + trueValues[i] * Math.log(predictedValues[i])))) );
        return loss.get() / trueValues.length;
    }
    private static double[] categoricalCrossEntropy(double[][] trueValues, double[][] predictedValues) {
        double [] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = categoricalCrossEntropy(trueValues[i], predictedValues[i]));
        return loss;
    }



    private static double contrastive(double[] dataPoints1, double[] dataPoints2, int y, double margin) {
        double distance = 0;
        for (int i = 0; i < dataPoints1.length; i++) {
            distance += (dataPoints1[i] - dataPoints2[i]) * (dataPoints1[i] - dataPoints2[i]);
        }
        distance = Math.sqrt(distance);

        double loss = (1 - y) * 0.5 * distance * distance +
                y * 0.5 * Math.pow(Math.max(0, margin - distance), 2);
        return loss;
    }
    private static double[] contrastive(double[][] trueValues, double[][] predictedValues, int y, double margin) {
        double[] loss = new double[trueValues.length]
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = contrastive(trueValues[i], predictedValues[i], y, margin));
        return loss;
    }


    private static double cosineProximity(double[] trueValues, double[] predictedValues) {
        if (trueValues.length != predictedValues.length) throw new IllegalArgumentException("Vectors must have the same dimension");

        double dotProduct = 0.0;
        double normYTrue = 0.0;
        double normYPred = 0.0;

        for (int i = 0; i < trueValues.length; i++) {
            dotProduct += trueValues[i] * predictedValues[i];
            normYTrue += trueValues[i] * predictedValues[i];
            normYPred += trueValues[i] * predictedValues[i];
        }

        double similarity = dotProduct / (Math.sqrt(normYTrue) * Math.sqrt(normYPred));
        return -similarity;  // Negative since we want to minimize the loss
    }
    private static double[] cosineProximity(double[][] trueValues, double[][] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = cosineProximity(trueValues[i], predictedValues[i]));
        return loss;
    }


    private static double focal(double trueValue, double predictedValue) { return 0; }
    private static double[] focal(double[] trueValues, double[] predictedValues) {}
    private static double[][] focal(double[][] trueValues, double[][] predictedValues) {}


    private static double hinge(double trueValue, double predictedValue) { return 0; }
    private static double[] hinge(double[] trueValues, double[] predictedValues) {}
    private static double[][] hinge(double[][] trueValues, double[][] predictedValues) {}


    private static double huber(double trueValue, double predictedValue) { return 0; }
    private static double[] huber(double[] trueValues, double[] predictedValues) {}
    private static double[][] huber(double[][] trueValues, double[][] predictedValues) {}


    private static double kullbackLeiblerDivergence(double trueValue, double predictedValue) { return 0; }
    private static double[] kullbackLeiblerDivergence(double[] trueValues, double[] predictedValues) {}
    private static double[][] kullbackLeiblerDivergence(double[][] trueValues, double[][] predictedValues) {}


    private static double logCosh(double trueValue, double predictedValue) { return 0; }
    private static double[] logCosh(double[] trueValues, double[] predictedValues) {}
    private static double[][] logCosh(double[][] trueValues, double[][] predictedValues) {}


    private static double absoluteError(double trueValue, double predictedValue) { return 0; }
    private static double[] absoluteError(double[] trueValues, double[] predictedValues) {}
    private static double[][] absoluteError(double[][] trueValues, double[][] predictedValues) {}


    private static double meanAbsolutePercentageError(double trueValue, double predictedValue) { return 0; }
    private static double[] meanAbsolutePercentageError(double[] trueValues, double[] predictedValues) {}
    private static double[][] meanAbsolutePercentageError(double[][] trueValues, double[][] predictedValues) {}


    private static double meanStandardError(double trueValue, double predictedValue) { return 0; }
    private static double[] meanStandardError(double[] trueValues, double[] predictedValues) {}
    private static double[][] meanStandardError(double[][] trueValues, double[][] predictedValues) {}


    private static double meanStandardLogarithmicError(double trueValue, double predictedValue) { return 0; }
    private static double[] meanStandardLogarithmicError(double[] trueValues, double[] predictedValues) {}
    private static double[][] meanStandardLogarithmicError(double[][] trueValues, double[][] predictedValues) {}


    private static double negativeLogLikelihood(double trueValue, double predictedValue) { return 0; }
    private static double[] negativeLogLikelihood(double[] trueValues, double[] predictedValues) {}
    private static double[][] negativeLogLikelihood(double[][] trueValues, double[][] predictedValues) {}


    private static double quadraticLoss(double trueValue, double predictedValue) { return 0; }
    private static double[] quadraticLoss(double[] trueValues, double[] predictedValues) {}
    private static double[][] quadraticLoss(double[][] trueValues, double[][] predictedValues) {}


    private static double rankNet(double trueValue, double predictedValue) { return 0; }
    private static double[] rankNet(double[] trueValues, double[] predictedValues) {}
    private static double[][] rankNet(double[][] trueValues, double[][] predictedValues) {}


    private static double sparseCategoricalCrossEntropy(double trueValue, double predictedValue) { return 0; }
    private static double[] sparseCategoricalCrossEntropy(double[] trueValues, double[] predictedValues) {}
    private static double[][] sparseCategoricalCrossEntropy(double[][] trueValues, double[][] predictedValues) {}


    private static double squaredHinge(double trueValue, double predictedValue) { return 0; }
    private static double[] squaredHinge(double[] trueValues, double[] predictedValues) {}
    private static double[][] squaredHinge(double[][] trueValues, double[][] predictedValues) {}


    private static double structuralSimilarityIndex(double trueValue, double predictedValue) { return 0; }
    private static double[] structuralSimilarityIndex(double[] trueValues, double[] predictedValues) {}
    private static double[][] structuralSimilarityIndex(double[][] trueValues, double[][] predictedValues) {}


    private static double tripletMargin(double trueValue, double predictedValue) { return 0; }
    private static double[] tripletMargin(double[] trueValues, double[] predictedValues) {}
    private static double[][] tripletMargin(double[][] trueValues, double[][] predictedValues) {}
}
