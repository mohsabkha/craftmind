package com.craftsentient.craftmind.errorLoss;

import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

public class ErrorLossFunctions {
    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double trueValues, double predictedValues) throws Exception {
        switch (lossFunction){
            case HINGE_LOSS_FUNCTION -> { return hinge(trueValues, predictedValues); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }

    public static Object lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[] trueValues, double[] predictedValues) throws Exception {
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> { return binaryCrossEntropy(trueValues, predictedValues); }
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return categoricalCrossEntropy(trueValues, predictedValues); }
            case COSINE_PROXIMITY_LOSS_FUNCTION -> { return cosineProximity(trueValues, predictedValues); }
            case HINGE_LOSS_FUNCTION -> { return hinge(trueValues, predictedValues); }
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
    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[] trueValues, double[] predictedValues, int y, double margin) throws Exception {
        switch (lossFunction){
            case CONTRASTIVE_LOSS_FUNCTION -> { return contrastive(trueValues, predictedValues, y, margin); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double[] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[][] trueValues, double[][] predictedValues, int y, double margin) throws Exception {
        switch (lossFunction){
            case CONTRASTIVE_LOSS_FUNCTION -> { return contrastive(trueValues, predictedValues, y, margin); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double trueValues, double predictedValues, double alpha, double gamma) throws Exception {
        switch (lossFunction){
            case FOCAL_LOSS_FUNCTION -> { return focal(trueValues, predictedValues, alpha, gamma); }

            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double[] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[] trueValues, double[] predictedValues, double[] alpha, double[] gamma) throws Exception {
        switch (lossFunction){
            case FOCAL_LOSS_FUNCTION -> { return focal(trueValues, predictedValues, alpha, gamma); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double trueValues, double predictedValues, double delta) throws Exception {
        switch (lossFunction){
            case HUBER_LOSS_FUNCTION -> { return huber(trueValues, predictedValues, delta); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double[] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[] trueValues, double[] predictedValues, double delta) throws Exception {
        switch (lossFunction){
            case HUBER_LOSS_FUNCTION -> { return huber(trueValues, predictedValues, delta); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double[][] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[][] trueValues, double[][] predictedValues, double delta) throws Exception {
        switch (lossFunction){
            case HUBER_LOSS_FUNCTION -> { return huber(trueValues, predictedValues, delta); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }


    private static double binaryCrossEntropy(double[] trueValues, double[] predictedValues) {
        AtomicReference<Double> loss = new AtomicReference<>((double) 0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss.updateAndGet( v ->
                (v + -trueValues[i] * Math.log(predictedValues[i]) - (1 - trueValues[i]) * Math.log(1 - predictedValues[i])))
        );
        return loss.get() / trueValues.length;
    }
    private static double[] binaryCrossEntropy(double[][] trueValues, double[][] predictedValues) {
        double [] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = binaryCrossEntropy(trueValues[i], predictedValues[i]));
        return loss;
    }


    private static double categoricalCrossEntropy(double[] trueValues, double[] predictedValues) {
        AtomicReference<Double> loss = new AtomicReference<>((double) 0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss.updateAndGet(v ->  v + trueValues[i] * Math.log(predictedValues[i])));
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
        return (1 - y) * 0.5 * distance * distance + y * 0.5 * Math.pow(Math.max(0, margin - distance), 2);

    }
    private static double[] contrastive(double[][] trueValues, double[][] predictedValues, int y, double margin) {
        double[] loss = new double[trueValues.length];
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


    private static double focal(double trueValue, double predictedValue, double alpha, double gamma) {
        double pt = (trueValue == 1) ? predictedValue : 1 - predictedValue;
        double alpha_t = (trueValue == 1) ? alpha : 1 - alpha;
        return -alpha_t * Math.pow(1 - pt, gamma) * Math.log(pt);
    }
    private static double[] focal(double[] trueValues, double[] predictedValues, double[] alpha, double[] gamma) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = focal(trueValues[i], predictedValues[i], alpha[i], gamma[i]));
        return loss;
    }
    private static double[][] focal(double[][] trueValues, double[][] predictedValues, double[][] alpha, double[][] gamma) {
        double[][] loss = new double[trueValues.length][];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = focal(trueValues[i], predictedValues[i], alpha[i], gamma[i]));
        return loss;
    }


    private static double hinge(double trueValue, double classifierOutput) {
        return Math.max(0, 1 - trueValue * classifierOutput);
    }
    private static double[] hinge(double[] trueValues, double[] classifierOutput) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = hinge(trueValues[i], classifierOutput[i]));
        return loss;
    }
    private static double[][] hinge(double[][] trueValues, double[][] classifierOutput) {
        double[][] loss = new double[trueValues.length][];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = hinge(trueValues[i], classifierOutput[i]));
        return loss;
    }


    private static double huber(double trueValue, double predictedValue, double delta) {
        double residual = trueValue - predictedValue;
        if (Math.abs(residual) <= delta) {
            // Quadratic loss for small residuals
            return 0.5 * Math.pow(residual, 2);
        } else {
            // Linear loss for large residuals
            return delta * (Math.abs(residual) - 0.5 * delta);
        }
    }
    private static double[] huber(double[] trueValues, double[] predictedValues, double delta) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = huber(trueValues[i], predictedValues[i], delta));
        return loss;
    }
    private static double[][] huber(double[][] trueValues, double[][] predictedValues, double delta) {
        double[][] loss = new double[trueValues.length][];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = huber(trueValues[i], predictedValues[i], delta));
        return loss;
    }


    private static double kullbackLeiblerDivergence(double[] trueValues, double[] predictedValues) {
        AtomicReference<Double> loss = new AtomicReference<>((double) 0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss.getAndUpdate( v -> v + trueValues[i] * Math.log(trueValues[i] / predictedValues[i])));
        return loss.get();
    }
    private static double[] kullbackLeiblerDivergence(double[][] trueValues, double[][] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = kullbackLeiblerDivergence(trueValues[i], predictedValues[i]));
        return loss;
    }


    private static double logCosh(double trueValue, double predictedValue) {
        double predictionError = trueValue - predictedValue;
        return Math.log(Math.cosh(predictionError));
    }
    private static double[] logCosh(double[] trueValues, double[] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = logCosh(trueValues[i], predictedValues[i]));
        return loss;
    }
    private static double[][] logCosh(double[][] trueValues, double[][] predictedValues) {
        double[][] loss = new double[trueValues.length][];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = logCosh(trueValues[i], predictedValues[i]));
        return loss;
    }


    /**
    * Compute the L1 loss.
    *
    * @param trueValues  Double array of true values.
    * @param predictedValues Double array of predicted values.
    * @return The L1 loss.
     */
    private static double absoluteError(double[] trueValues, double[] predictedValues) {
        double loss = 0.0;
        for (int i = 0; i < trueValues.length; i++) {
            loss += Math.abs(trueValues[i] - predictedValues[i]);
        }
        return loss / trueValues.length;
    }
    private static double[] absoluteError(double[][] trueValues, double[][] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = absoluteError(trueValues[i], predictedValues[i]));
        return loss;
    }


    private static double meanAbsolutePercentageError(double[] trueValues, double[] predictedValues) {
        double mape = 0.0;
        for (int i = 0; i < trueValues.length; i++) {
            if (trueValues[i] == 0) {
                throw new IllegalArgumentException("True values should not be zero, as this would cause division by zero in MAPE.");
            }
            mape += Math.abs((trueValues[i] - predictedValues[i]) / trueValues[i]);
        }
        return (mape / trueValues.length) * 100.0;  // Multiply by 100 to get a percentage
    }
    private static double[] meanAbsolutePercentageError(double[][] trueValues, double[][] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = meanAbsolutePercentageError(trueValues[i], predictedValues[i]));
        return loss;
    }


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
