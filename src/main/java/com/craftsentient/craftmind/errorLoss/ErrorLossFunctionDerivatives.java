package com.craftsentient.craftmind.errorLoss;

import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

public class ErrorLossFunctionDerivatives {
    // Binary Cross Entropy
    public static double binaryCrossEntropy(double trueValue, double predictedValueProbability){
        return -trueValue / predictedValueProbability + (1 - trueValue) / (1 - predictedValueProbability);
    }
    public static double binaryCrossEntropy(double[] trueValues, double[] predictedValueProbabilities) {
        AtomicReference<Double> loss = new AtomicReference<>((double) 0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss.updateAndGet(v -> (v + binaryCrossEntropy(trueValues[i], predictedValueProbabilities[i]))));
        return loss.get() / trueValues.length;
    }
    public static double[] binaryCrossEntropy(double[][] trueValues, double[][] predictedValues) {
        double [] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = binaryCrossEntropy(trueValues[i], predictedValues[i]));
        return loss;
    }

    // Categorical Cross Entropy
    public static double categoricalCrossEntropy(double trueValues, double predictedProbability){
        return -trueValues/predictedProbability;
    }
    public static double[] categoricalCrossEntropy(double[] trueValues, double[] predictedValues) {
        if(trueValues.length < 1) { throw new IllegalArgumentException("True Values must at least have 1 value when passed to categorical cross entropy"); }
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = categoricalCrossEntropy(trueValues[i], predictedValues[i]));
        return loss;
    }
    public static double[][] categoricalCrossEntropy(double[][] trueValues, double[][] predictedValues) {
        double [][] loss = new double[trueValues.length][];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = categoricalCrossEntropy(trueValues[i], predictedValues[i]));
        return loss;
    }

    // Contrastive, y = binary label (0 for similar, 1 for dissimilar)
    public static double contrastive(double y, double distance){
        if(y == 0) {
            return distance;
        } else {
            double margin = 1;
            return distance >= margin ? 0 : -(margin - distance);
        }
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
    // Contrastive, y = binary label (0 for similar, 1 for dissimilar)
    public static double contrastiveWithTwoDatasets(double[] dataPoints1, double[] dataPoints2, int y, double margin) {
        double distance = 0;
        for (int i = 0; i < dataPoints1.length; i++) {
            distance += (dataPoints1[i] - dataPoints2[i]) * (dataPoints1[i] - dataPoints2[i]);
        }
        distance = Math.sqrt(distance);
        if(y == 0) {
            return distance;
        } else {
            return distance >= margin? 0 : -(margin - distance);
        }
    }
    public static double[] contrastiveWithTwoDatasets(double[][] trueValues, double[][] predictedValues, int y, double margin) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = contrastiveWithTwoDatasets(trueValues[i], predictedValues[i], y, margin));
        return loss;
    }


    // TODO: Learn and take derivative of this function later
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
        return -alpha_t * gamma * Math.pow(1 - pt, gamma - 1) * (Math.log(pt) + (1 - pt));
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


    // Hinge Loss
    private static double hinge(double trueValue, double classifierOutput) {
        if((trueValue * classifierOutput) >= 1){
            return 0;
        } else {
            return -trueValue;
        }
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


    // Huber
    private static double huber(double trueValue, double predictedValue, double delta_threshold) {
        double residual = trueValue - predictedValue;
        if(Math.abs(residual) <= delta_threshold){
            return -residual;
        } else if (residual > delta_threshold){
            return -delta_threshold;
        } else {
            return  delta_threshold;
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


    //.TODO: make the derivative of this and also implement auto derivative functions
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
        return -Math.tanh(predictionError);
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


    public static double absoluteError(double trueValue, double predictedValue){
        double predictionError = trueValue - predictedValue;
        if(predictionError > 0) {
            return -1;
        } else if(predictionError < 0) {
            return 1;
        }
        else {
            throw new RuntimeException("Absolute Error Loss Function Derivative Cannot Be Derived For Values between -1 and 1");
        }
    }
    /**
     * Compute the L1 loss.
     *
     * @param trueValues  Double array of true values.
     * @param predictedValues Double array of predicted values.
     * @return The L1 loss.
     */
    private static double[] absoluteError(double[] trueValues, double[] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = absoluteError(trueValues[i], predictedValues[i]));
        return loss;
    }
    private static double[][] absoluteError(double[][] trueValues, double[][] predictedValues) {
        double[][] loss = new double[trueValues.length][];
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
            if(trueValues[i] - predictedValues[i] > 0){
                mape += (-1/trueValues[i]);
            } else if (trueValues[i] - predictedValues[i] < 0){
                mape += (1/trueValues[i]);
            }
        }
        return (mape / trueValues.length) * 100.0;  // Multiply by 100 to get a percentage
    }
    private static double[] meanAbsolutePercentageError(double[][] trueValues, double[][] predictedValues) {
        double[] loss = new double[trueValues.length];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> loss[i] = meanAbsolutePercentageError(trueValues[i], predictedValues[i]));
        return loss;
    }


    public static double meanStandardLogarithmicError(double trueValue, double predictedValue){
        return -2*((Math.log1p(trueValue) - Math.log1p(predictedValue))/(1 + predictedValue));
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
        return msle/trueValues.length;
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
    private static double[] negativeLogLikelihood(int trueClass, double[] predictedValues) {
        double[] gradient = new double[predictedValues.length];
        gradient[trueClass] = -1.0 / predictedValues[trueClass];
        return gradient;
    }
    private static double[][] negativeLogLikelihood(int[] trueValues, double[] predictedValues){
        double[][] loss = new double[trueValues.length][];
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
    private static double[][] negativeLogLikelihood(int[] trueValues, double[][] predictedValues) {
        double[][] loss = new double[trueValues.length][];
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
    private static double[][] negativeLogLikelihood(int[][] oneHotEncodedVec, double[][] predictedValues) {
        double[][] loss = new double[oneHotEncodedVec.length][];
        int[] trueClasses = new int[oneHotEncodedVec.length];
        IntStream.range(0, oneHotEncodedVec.length).parallel().forEachOrdered(i -> {
            IntStream.range(0, oneHotEncodedVec[i].length).parallel().forEachOrdered(j -> {
                if(oneHotEncodedVec[i][j] == 1) trueClasses[i] = j;
            });
        });
        IntStream.range(0, trueClasses.length).parallel().forEachOrdered(i -> loss[i] = negativeLogLikelihood(trueClasses[i], predictedValues[i]));
        return loss;
    }


    public static double quadratic(double trueValue, double predictedValue){
        return 2 * (predictedValue - trueValue);
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
            totalSquaredError += quadratic(trueValues[i], predictedValues[i]);
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
    public static double rankNetClassifier1(double classifier1, double classifier2, int label1, int label2, double sigma) {
        double o_ij = classifier1 - classifier2;
        double P_ij = 1.0 / (1.0 + Math.exp(-sigma * o_ij));

        double S_ij;
        if (label1 > label2) {
            S_ij = 1.0;
        } else if (label1 < label2) {
            S_ij = 0.0;
        } else {
            S_ij = 0.5;
        }

        return sigma * (P_ij - S_ij);
    }
    private static double[] rankNetClassifier1(double[] classifier1, double[] classifier2, int[] label1, int[] label2, double[] sigma) {
        double[] loss = new double[classifier1.length];
        IntStream.range(0, classifier1.length).parallel().forEachOrdered(i -> loss[i] = rankNetClassifier1(classifier1[i], classifier2[i], label1[i], label2[i], sigma[i]));
        return loss;
    }
    private static double[][] rankNetClassifier1(double[][] classifier1, double[][] classifier2, int[][] label1, int[][] label2, double[][] sigma) {
        double[][] loss = new double[classifier1.length][];
        IntStream.range(0, classifier1.length).parallel().forEachOrdered(i -> loss[i] = rankNetClassifier1(classifier1[i], classifier2[i], label1[i], label2[i], sigma[i]));
        return loss;
    }
    public static double rankNetClassifier2(double classifier1, double classifier2, int label1, int label2, double sigma){
        return -rankNetClassifier1(classifier1, classifier2, label1, label2, sigma);
    }
    private static double[] rankNetClassifier2(double[] classifier1, double[] classifier2, int[] label1, int[] label2, double[] sigma) {
        double[] loss = new double[classifier1.length];
        IntStream.range(0, classifier1.length).parallel().forEachOrdered(i -> loss[i] = rankNetClassifier2(classifier1[i], classifier2[i], label1[i], label2[i], sigma[i]));
        return loss;
    }
    private static double[][] rankNetClassifier2(double[][] classifier1, double[][] classifier2, int[][] label1, int[][] label2, double[][] sigma) {
        double[][] loss = new double[classifier1.length][];
        IntStream.range(0, classifier1.length).parallel().forEachOrdered(i -> loss[i] = rankNetClassifier2(classifier1[i], classifier2[i], label1[i], label2[i], sigma[i]));
        return loss;
    }


    /**
     * Compute the Sparse Categorical Cross-Entropy loss.
     *
     * @param predictedValues Array of predicted probabilities for each class.
     * @param trueValueIndex The index of the true class for the sample.
     * @return The Sparse Categorical Cross-Entropy loss for the given predictions and true class.
     */
    private static double[] sparseCategoricalCrossEntropy(int trueValueIndex, double[] predictedValues) {
        if (trueValueIndex < 0 || trueValueIndex >= predictedValues.length) {
            throw new IllegalArgumentException("Invalid trueClassIndex");
        }
        double[] gradient = new double[predictedValues.length];

        for (int i = 0; i < predictedValues.length; i++) {
            if (i == trueValueIndex) {
                gradient[i] = -1.0 / predictedValues[trueValueIndex];
            } else {
                gradient[i] = 0;
            }
        }

        return gradient;
    }
    private static double[][] sparseCategoricalCrossEntropy(int[] trueValues, double[][] predictedValues) {
        double[][] gradients = new double[trueValues.length][];
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> gradients[i] = sparseCategoricalCrossEntropy(trueValues[i], predictedValues[i]));
        return gradients;
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
