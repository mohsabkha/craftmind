package com.craftsentient.craftmind.errorLoss;

import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

public class ErrorLossFunctions {
    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double trueValues, double output) throws Exception {
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> { return binaryCrossEntropy(trueValues, output); }
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return categoricalCrossEntropy(trueValues, output); }
            case HINGE_LOSS_FUNCTION -> { return hinge(trueValues, output); }
            case LOG_COSH_LOSS_FUNCTION -> { return logCosh(trueValues, output); }
            case MSLE_LOSS_FUNCTION -> {  return meanStandardLogarithmicError(trueValues, output); }
            case SQUARED_HINGE_LOSS_FUNCTION -> { return squaredHinge(trueValues, output); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }

    public static Object lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[] trueValues, double[] predictedValues) throws Exception {
        if(trueValues.length != predictedValues.length) throw new IllegalArgumentException("Number of true values and predicted values must be the same!");
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> { return binaryCrossEntropy(trueValues, predictedValues); }
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return categoricalCrossEntropy(trueValues, predictedValues); }
            case COSINE_PROXIMITY_LOSS_FUNCTION -> { return cosineProximity(trueValues, predictedValues); }
            case HINGE_LOSS_FUNCTION -> { return hinge(trueValues, predictedValues); }
            case KL_DIVERGENCE_LOSS_FUNCTION -> { return kullbackLeiblerDivergence(trueValues, predictedValues); }
            case LOG_COSH_LOSS_FUNCTION -> { return logCosh(trueValues, predictedValues); }
            case L1_LOSS_FUNCTION -> { return absoluteError(trueValues, predictedValues); }
            case MAPE_LOSS_FUNCTION -> { return meanAbsolutePercentageError(trueValues, predictedValues); }
            //case MSE_LOSS_FUNCTION -> { return meanStandardError(trueValues, predictedValues); }
            case MSLE_LOSS_FUNCTION -> {  return meanStandardLogarithmicError(trueValues, predictedValues); }
            case QUADRATIC_LOSS -> { return quadratic(trueValues, predictedValues); }
            //case RANKNET_LOSS_FUNCTION -> { return rankNet(trueValues, predictedValues); }
            //case SPARSE_CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return sparseCategoricalCrossEntropy(trueValues, predictedValues); }
            case SQUARED_HINGE_LOSS_FUNCTION -> { return squaredHinge(trueValues, predictedValues); }
            //case SSIM_LOSS_FUNCTION -> { return structuralSimilarityIndex(trueValues, predictedValues); }
            //case TRIPLET_MARGIN_LOSS_FUNCTION -> { return tripletMargin(trueValues, predictedValues); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static Object lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[][] trueValues, double[][] predictedValues) throws Exception {
        if(trueValues.length != predictedValues.length) throw new IllegalArgumentException("Number of true values and predicted values must be the same!");
        switch (lossFunction){
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION -> { return binaryCrossEntropy(trueValues, predictedValues); }
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return categoricalCrossEntropy(trueValues, predictedValues); }
            case COSINE_PROXIMITY_LOSS_FUNCTION -> { return cosineProximity(trueValues, predictedValues); }
            case KL_DIVERGENCE_LOSS_FUNCTION -> { return kullbackLeiblerDivergence(trueValues, predictedValues); }
            case LOG_COSH_LOSS_FUNCTION -> { return logCosh(trueValues, predictedValues); }
            case L1_LOSS_FUNCTION -> { return absoluteError(trueValues, predictedValues); }
            case MAPE_LOSS_FUNCTION -> { return meanAbsolutePercentageError(trueValues, predictedValues); }
            //case MSE_LOSS_FUNCTION -> { return meanStandardError(trueValues, predictedValues); }
            case MSLE_LOSS_FUNCTION -> {  return meanStandardLogarithmicError(trueValues, predictedValues); }
            case QUADRATIC_LOSS -> { return quadratic(trueValues, predictedValues); }
            //case RANKNET_LOSS_FUNCTION -> { return rankNet(trueValues, predictedValues); }
            //case SPARSE_CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION -> { return sparseCategoricalCrossEntropy(trueValues, predictedValues); }
            case SQUARED_HINGE_LOSS_FUNCTION -> { return squaredHinge(trueValues, predictedValues); }
            //case SSIM_LOSS_FUNCTION -> { return structuralSimilarityIndex(trueValues, predictedValues); }
            //case TRIPLET_MARGIN_LOSS_FUNCTION -> { return tripletMargin(trueValues, predictedValues); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }

    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueClass, double[] predictedValues) throws Exception {
        switch (lossFunction) {
            case NLL_LOSS_FUNCTION -> { return negativeLogLikelihood(trueClass, predictedValues); }
            default -> {
                throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name() + ". Only Available Loss Function For Data Entered Is NLL_LOSS_FUNCTION");
            }
        }
    }
    public static double[] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int[] trueClass, double[][] predictedValues) throws Exception {
        if(trueClass.length != predictedValues.length) throw new IllegalArgumentException("Number of true values and predicted values must be the same!");
        switch (lossFunction) {
            case NLL_LOSS_FUNCTION -> { return negativeLogLikelihood(trueClass, predictedValues); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double[] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int[] trueClass, double[] predictedValues) throws Exception {
        if(trueClass.length != predictedValues.length) throw new IllegalArgumentException("Number of true values and predicted values must be the same!");
        switch (lossFunction) {
            case NLL_LOSS_FUNCTION -> { return negativeLogLikelihood(trueClass, predictedValues); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double[] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int[][] trueClass, double[][] predictedValues) throws Exception {
        if(trueClass.length != predictedValues.length) throw new IllegalArgumentException("Number of true values and predicted values must be the same!");
        switch (lossFunction) {
            case NLL_LOSS_FUNCTION -> { return negativeLogLikelihood(trueClass, predictedValues); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }

    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[] trueValues, double[] predictedValues, int y, double margin) throws Exception {
        if(trueValues.length != predictedValues.length) throw new IllegalArgumentException("Number of true values and predicted values must be the same!");
        switch (lossFunction){
            case CONTRASTIVE_LOSS_FUNCTION -> { return contrastiveWithTwoDataSets(trueValues, predictedValues, y, margin); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double[] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[][] trueValues, double[][] predictedValues, int y, double margin) throws Exception {
        if(trueValues.length != predictedValues.length) throw new IllegalArgumentException("Number of true values and predicted values must be the same!");
        switch (lossFunction){
            case CONTRASTIVE_LOSS_FUNCTION -> { return contrastiveWithTwoDataSets(trueValues, predictedValues, y, margin); }
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
        if(trueValues.length != predictedValues.length) throw new IllegalArgumentException("Number of true values and predicted values must be the same!");
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
        if(trueValues.length != predictedValues.length) throw new IllegalArgumentException("Number of true values and predicted values must be the same!");
        switch (lossFunction){
            case HUBER_LOSS_FUNCTION -> { return huber(trueValues, predictedValues, delta); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }
    public static double[][] lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, double[][] trueValues, double[][] predictedValues, double delta) throws Exception {
        if(trueValues.length != predictedValues.length) throw new IllegalArgumentException("Number of true values and predicted values must be the same!");
        switch (lossFunction){
            case HUBER_LOSS_FUNCTION -> { return huber(trueValues, predictedValues, delta); }
            default -> throw new Exception("Incorrect Loss Function Name Entered: " + lossFunction.name());
        }
    }






    private static double binaryCrossEntropy(double trueValue, double predictedValueProbability){
        return -trueValue * Math.log(predictedValueProbability) - (1 - trueValue) * Math.log(1 - predictedValueProbability);
    }
    private static double binaryCrossEntropy(double[] trueValues, double[] predictedValueProbabilities) {
        AtomicReference<Double> loss = new AtomicReference<>((double) 0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss.updateAndGet( v -> ( v + binaryCrossEntropy(trueValues[i], predictedValueProbabilities[i])) ));
        return loss.get() / trueValues.length;
    }
    private static double[] binaryCrossEntropy(double[][] trueValues, double[][] predictedValueProbabilities) {
        double [] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = binaryCrossEntropy(trueValues[i], predictedValueProbabilities[i]));
        return loss;
    }


    private static double categoricalCrossEntropy(double trueValues, double predictedProbability){
        return trueValues * Math.log(predictedProbability);
    }
    private static double categoricalCrossEntropy(double[] trueValues, double[] predictedValueProbabilities) {
        if(trueValues.length < 1) { throw new IllegalArgumentException("True Values must at least have 1 value when passed to categorical cross entropy"); }
        AtomicReference<Double> loss = new AtomicReference<>((double) 0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss.updateAndGet(v ->  v - categoricalCrossEntropy(trueValues[i], predictedValueProbabilities[i])));
        return loss.get() / trueValues.length;
    }
    private static double[] categoricalCrossEntropy(double[][] trueValues, double[][] predictedValueProbabilities) {
        double [] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = categoricalCrossEntropy(trueValues[i], predictedValueProbabilities[i]));
        return loss;
    }


    // Contrastive, y = binary label (0 for similar, 1 for dissimilar)
    public static double contrastive(double y, double distance){
        double margin = 1;
        return (1 - y) * 0.5 * distance * distance + y * 0.5 * Math.pow(Math.max(0, margin - distance), 2);
    }
    public static double[] contrastive(double[] y, double[] distances){
        double[] loss = new double[distances.length];
        double margin = 1;
        IntStream.range(0, loss.length).parallel().forEachOrdered(i -> loss[i] = contrastive(y[i], distances[i]));
        return loss;
    }
    public static double[][] contrastive(double[][] y, double[][] distances){
        double[][] loss = new double[distances.length][];
        double margin = 1;
        IntStream.range(0, loss.length).parallel().forEachOrdered(i -> loss[i] = contrastive(y[i], distances[i]));
        return loss;
    }
    private static double contrastiveWithTwoDataSets(double[] dataPoints1, double[] dataPoints2, int y, double margin) {
        double distance = 0;
        for (int i = 0; i < dataPoints1.length; i++) {
            distance += (dataPoints1[i] - dataPoints2[i]) * (dataPoints1[i] - dataPoints2[i]);
        }
        distance = Math.sqrt(distance);
        return (1 - y) * 0.5 * distance * distance + y * 0.5 * Math.pow(Math.max(0, margin - distance), 2);
    }
    private static double[] contrastiveWithTwoDataSets(double[][] trueValues, double[][] predictedValues, int y, double margin) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = contrastiveWithTwoDataSets(trueValues[i], predictedValues[i], y, margin));
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


    /*
     * Used for object detection
     * alpha is a balancing factor
     * gamma is the focusing parameter to reduce contributions to easy examples
     */
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
        AtomicReference<Double> loss = new AtomicReference<>((double) 0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss.getAndUpdate( v -> v + Math.abs(trueValues[i] - predictedValues[i])));
        return loss.get();
    }
    private static double[] absoluteError(double[][] trueValues, double[][] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = absoluteError(trueValues[i], predictedValues[i]));
        return loss;
    }

    /**
     * Compute the Mean Absolute Percentage Error (MAPE).
     *
     * @param trueValues  Array of true values.
     * @param predictedValues Array of predicted values.
     * @return The MAPE value.
     */
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


    public static double meanStandardLogarithmicError(double trueValue, double predictedValue){
        return Math.pow(Math.log1p(trueValue) - Math.log1p(predictedValue), 2);
    }
    /**
     * Compute the Mean Squared Logarithmic Error (MSLE).
     *
     * @param trueValues  Array of true values.
     * @param predictedValues Array of predicted values.
     * @return The MSLE value.
     */
    private static double meanStandardLogarithmicError(double[] trueValues, double[] predictedValues) {
        double msle = 0.0;
        for (int i = 0; i < trueValues.length; i++) {
            if (trueValues[i] < 0 || predictedValues[i] < 0) {
                throw new IllegalArgumentException("Values should not be negative, as this would cause issues with logarithms in MSLE.");
            }
            msle += meanStandardLogarithmicError(trueValues[i], predictedValues[i]);
        }
        return msle;
    }
    private static double[] meanStandardLogarithmicError(double[][] trueValues, double[][] predictedValues) {
            double[] loss = new double[trueValues.length];
            IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = meanStandardLogarithmicError(trueValues[i], predictedValues[i]));
            return loss;
    }

    /**
     * Compute the Negative Log Likelihood (NLL) Loss for a single observation.
     *
     * @param predictedValues Array of predicted probabilities for each class.
     * @param trueClass The actual class index.
     * @return The NLL value for the given observation.
     */
    private static double negativeLogLikelihood(int trueClass, double[] predictedValues) {
        double predictedProbability = predictedValues[trueClass];
        if (predictedProbability <= 0 || predictedProbability > 1) {
            throw new IllegalArgumentException("Incompatible loss function and activation function! Values for predictions should be between 0 and 1!");
        }
        return -Math.log(predictedProbability);
    }
    private static double[] negativeLogLikelihood(int[] trueValues, double[] predictedValues){
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i ->
                loss[i] = negativeLogLikelihood(trueValues[i], predictedValues));
        return loss;
    }
    /**
     * Compute the Negative Log Likelihood (NLL) Loss for a series of observations.
     *
     * @param predictedValues Array of predicted batch probabilities for each class.
     * @param trueValues The actual class indices.
     * @return The NLL value for the given observation.
     */
    private static double[] negativeLogLikelihood(int[] trueValues, double[][] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> {
            loss[i] = negativeLogLikelihood(trueValues[i], predictedValues[i]);
        });
        return loss;
    }
    /**
     * Compute the Negative Log Likelihood (NLL) Loss for a batch series of observations using one hot encoded vectors
     *
     * @param predictedValues Array of predicted probabilities for each class.
     * @param oneHotEncodedVec One hot encoded vectors - array of arrays, where inner array has 1 for any position that is correct and 0 for incorrect
     * @return The NLL value for the given observation.
     */
    private static double[] negativeLogLikelihood(int[][] oneHotEncodedVec, double[][] predictedValues) {
        double[] loss = new double[oneHotEncodedVec.length];
        int[] trueClasses = new int[oneHotEncodedVec.length];
        IntStream.range(0, oneHotEncodedVec.length).parallel().forEachOrdered(i -> {
            IntStream.range(0, oneHotEncodedVec[i].length).parallel().forEachOrdered(j -> {
                if(oneHotEncodedVec[i][j] == 1) trueClasses[i] = j;
            });
        });
        IntStream.range(0, trueClasses.length).parallel().forEachOrdered(i -> loss[i] = negativeLogLikelihood(trueClasses[i], predictedValues[i]));
        return loss;
    }


    /**
     * Compute the Quadratic (Mean Squared Error) Loss.
     *
     * @param trueValues Array of true values.
     * @param predictedValues Array of predicted values.
     * @return The Quadratic Loss for the given values.
     */
    private static double quadratic(double[] trueValues, double[] predictedValues) {
        double totalSquaredError = 0.0;
        for (int i = 0; i < trueValues.length; i++) {
            double difference = trueValues[i] - predictedValues[i];
            totalSquaredError += difference * difference;
        }
        return totalSquaredError / trueValues.length;
    }
    private static double[] quadratic(double[][] trueValues, double[][] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = quadratic(trueValues[i], predictedValues[i]));
        return loss;
    }


    /**
     * Compute the RankNet pairwise loss.
     *
     * @param classifier1 Score for document i.
     * @param classifier2 Score for document j.
     * @param label1 Relevance label for document i.
     * @param label2 Relevance label for document j.
     * @param sigma Constant determining the steepness of the logistic function.
     * @return The RankNet loss for the given scores and relevance labels.
     */
    private static double rankNet(double classifier1, double classifier2, int label1, int label2, double sigma) {
        double o_ij = classifier1- classifier1;
        double P_ij = 1.0 / (1.0 + Math.exp(-sigma * o_ij));

        double P_bar_ij;
        if (label1 > label2) { P_bar_ij = 1.0; }
        else if (label1 < label2) { P_bar_ij = 0.0; }
        else { P_bar_ij = 0.5; }
        return -P_bar_ij * Math.log(P_ij) - (1.0 - P_bar_ij) * Math.log(1.0 - P_ij);
    }
    private static double[] rankNet(double[] classifier1, double[] classifier2, int[] label1, int[] label2, double[] sigma) {
        double[] loss = new double[classifier1.length];
        IntStream.range(0, classifier1.length).parallel().forEachOrdered(i -> loss[i] = rankNet(classifier1[i], classifier2[i], label1[i], label2[i], sigma[i]));
        return loss;
    }
    private static double[][] rankNet(double[][] classifier1, double[][] classifier2, int[][] label1, int[][] label2, double[][] sigma) {
        double[][] loss = new double[classifier1.length][];
        IntStream.range(0, classifier1.length).parallel().forEachOrdered(i -> loss[i] = rankNet(classifier1[i], classifier2[i], label1[i], label2[i], sigma[i]));
        return loss;
    }



    /**
     * Compute the Sparse Categorical Cross-Entropy loss.
     *
     * @param predictedValues Array of predicted probabilities for each class.
     * @param trueValueIndex The index of the true class for the sample.
     * @return The Sparse Categorical Cross-Entropy loss for the given predictions and true class.
     */
    private static double sparseCategoricalCrossEntropy(int trueValueIndex, double[] predictedValues) {
        if (trueValueIndex < 0 || trueValueIndex >= predictedValues.length) {
            throw new IllegalArgumentException("Invalid trueClassIndex");
        }
        return -Math.log(predictedValues[trueValueIndex]);
    }
    private static double[] sparseCategoricalCrossEntropy(int[] trueValues, double[][] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = sparseCategoricalCrossEntropy(trueValues[i], predictedValues[i]));
        return loss;
    }


    /**
     * Compute the squared hinge loss.
     *
     * @param trueValue True label (+1 or -1).
     * @param predictedValue Raw prediction value.
     * @return The squared hinge loss for the given true label and prediction.
     */
    private static double squaredHinge(double trueValue, double predictedValue) {
        double loss = Math.max(0, 1 - trueValue * predictedValue);
        return loss * loss;
    }
    private static double[] squaredHinge(double[] trueValues, double[] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = squaredHinge(trueValues[i], predictedValues[i]));
        return loss;
    }
    private static double[][] squaredHinge(double[][] trueValues, double[][] predictedValues) {
        double[][] loss = new double[trueValues.length][];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = squaredHinge(trueValues[i], predictedValues[i]));
        return loss;
    }


    private static double computeMean(double[] data){
        double sum = 0;
        for (double value : data) { sum += value; }
        return sum / data.length;
    }
    private static double computeVariance(double[] data) {
        double mean = computeMean(data);
        double temp = 0;
        for (double value : data) {
            temp += (value - mean) * (value - mean);
        }
        return temp / data.length;
    }
    private static double computeLossForChannel(double[] firstImageChannel, double[] secondImageChannel) {
        // Compute means
        double muX = computeMean(firstImageChannel);
        double muY = computeMean(secondImageChannel);

        // Compute variances
        double sigmaX2 = computeVariance(firstImageChannel);
        double sigmaY2 = computeVariance(secondImageChannel);

        // Compute covariance
        double sigmaXY = 0;
        for (int i = 0; i < firstImageChannel.length; i++) { sigmaXY += (firstImageChannel[i] - muX) * (secondImageChannel[i] - muY); }
        sigmaXY /= firstImageChannel.length - 1;

        // Constants for stability
        double c1 = 0.01 * 0.01;
        double c2 = 0.03 * 0.03;

        // Compute SSIM for this channel
        double ssim = ((2 * muX * muY + c1) * (2 * sigmaXY + c2)) /
                ((muX * muX + muY * muY + c1) * (sigmaX2 + sigmaY2 + c2));

        // Return loss for this channel
        return 1 - ssim;
    }
    private static double structuralSimilarityIndex(double[][][] image1, double[][][] image2) {
        if (image1.length != image2.length || image1[0].length != image2[0].length || image1[0][0].length != image2[0][0].length) {
            throw new IllegalArgumentException("Images must have the same dimensions");
        }
        int height = image1.length;
        int width = image1[0].length;
        int channels = image1[0][0].length;
        double totalLoss = 0;
        for (int c = 0; c < channels; c++) {
            double[] flatImage1 = new double[height * width];
            double[] flatImage2 = new double[height * width];
            int k = 0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    flatImage1[k] = image1[i][j][c];
                    flatImage2[k] = image2[i][j][c];
                    k++;
                }
            }
            totalLoss += computeLossForChannel(flatImage1, flatImage2);
        }
        return totalLoss / channels; // Average loss across channels
    }


    private static double distance(double[] vec1, double[] vec2) {
        double sum = 0.0;
        for (int i = 0; i < vec1.length; i++) { sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]); }
        return Math.sqrt(sum);
    }
    private static double tripletMargin(double[] anchor, double[] positive, double[] negative, double margin) {
        double posDist = distance(anchor, positive);
        double negDist = distance(anchor, negative);
        return Math.max(posDist - negDist + margin, 0);
    }
    private static double[] tripletMargin(double[][] anchors, double[][] positives, double[][] negatives, double[] margins) {
        double[] loss = new double[anchors.length];
        IntStream.range(0, anchors.length).parallel().forEachOrdered(i -> loss[i] = tripletMargin(anchors[i], positives[i], negatives[i], margins[i]));
        return loss;
    }
}
