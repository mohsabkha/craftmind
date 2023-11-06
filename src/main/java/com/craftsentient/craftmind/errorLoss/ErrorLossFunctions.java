package com.craftsentient.craftmind.errorLoss;

import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

import static com.craftsentient.craftmind.layers.DenseLayers.*;
import static com.craftsentient.craftmind.utils.PrintUtils.printGeneric;
import static com.craftsentient.craftmind.utils.PrintUtils.printInfo;

public class ErrorLossFunctions {

    public static double lossFunction(DEFAULT_LOSS_FUNCTIONS lossFunction, int trueValueIndex, int selectedOutputIndex, double[] outputs){
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


    /*
     * For binary classification
     * Compatible with most activation functions. Please use unbounded activation functions in output layer though
     */
    private static double binaryCrossEntropy(int trueValueIndex, int selectedOutputIndex, double[] outputs){
        return -outputs[trueValueIndex] * Math.log(outputs[selectedOutputIndex]) - (1 - outputs[trueValueIndex]) * Math.log(1 - outputs[selectedOutputIndex]);
    }

    /*
     * For binary classification
     * For this, the output should ideally be a -1 or 1, but passing in the index will auto convert to these values
     */
    private static double binaryHinge(int trueValueIndex, int selectedOutputIndex, double[] outputs){
        double margin = 0.5;
        int trueLabel = trueValueIndex * 2 - 1;
        int selectedLabel = selectedOutputIndex * 2 - 1;

        return Math.max(0, margin - trueLabel * selectedLabel);
    }

    /*
     * For multi-class classification
     * Compatible with most activation functions. Please use unbounded activation functions in output layer though
     */
    private static double categoricalCrossEntropy(int trueValueIndex, double[] outputs){
        return -Math.log(outputs[trueValueIndex]);
    }

    /*
     * For two concurrent Siamese networks
     * Compatible with most activation functions. Please use unbounded activation functions in output layer though
     */
    public static double contrastive(double firstSiameseValue, double secondSiameseValue){
        double distance = firstSiameseValue - secondSiameseValue;
        double y;
        if(distance < MARGIN) {
            y = 0;
        } else {
            y = 1;
        }
        return (1 - y) * 0.5 * distance * distance + y * 0.5 * Math.pow(Math.max(0, MARGIN - distance), 2);
    }

    private static double hinge(int trueValueIndex, double[] outputs){
        double trueScore = outputs[trueValueIndex];
        double maxIncorrectScore = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < outputs.length; i++) {
            if (i != trueValueIndex) {
                maxIncorrectScore = Math.max(maxIncorrectScore, outputs[i]);
            }
        }

        return Math.max(0, 1 + maxIncorrectScore - trueScore);
    }

    private static double logCosh(int trueValueIndex, double[] outputs) {
        // The true probability for the correct class is 1, and 0 for all other classes
        double trueValue = 1.0;
        // Get the predicted probability for the true class
        double predictedValue = outputs[trueValueIndex];
        // Calculate the prediction error for the true class
        double predictionError = predictedValue - trueValue;
        // Calculate and return the log-cosh of the prediction error
        return Math.log(Math.cosh(predictionError));
    }

    public static double meanStandardLogarithmicError(int trueValueIndex, int selectedOutputIndex, double[] outputs){
        return Math.pow(Math.log1p(outputs[trueValueIndex]) - Math.log1p(outputs[selectedOutputIndex]), 2);
    }

    private static double squaredHinge(int trueValueIndex, double[] outputs) {
        double loss = 0.0;
        for (int i = 0; i < outputs.length; i++) {
            if (i == trueValueIndex) {
                // For the true class, we subtract the output from 1.
                loss += Math.pow(Math.max(0, 1 - outputs[i]), 2);
            } else {
                // For the incorrect classes, we add 1 to the output.
                loss += Math.pow(Math.max(0, 1 + outputs[i]), 2);
            }
        }
        return loss;
    }

    /*
     * Used for object detection
     * alpha is a balancing factor
     * gamma is the focusing parameter to reduce contributions to easy examples
     */
    private static double focal(int trueValueIndex, double[] outputs) {
        // Extract the predicted probability associated with the true class
        double p = outputs[trueValueIndex];
        // Calculate the modulating factor (1 - p_t)^gamma
        double modulatingFactor = Math.pow(1 - p, GAMMA);
        // Calculate the focal loss
        return -ALPHA * modulatingFactor * Math.log(p);
    }



    // Contrastive, y = binary label (0 for similar, 1 for dissimilar)
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


    private static double huber(int trueValueIndex, double[] outputs) {
        double loss = 0.0;
        for (int i = 0; i < outputs.length; i++) {
            // Convert true value index to one-hot encoding, 1 for true class, 0 for all others
            double trueValue = (i == trueValueIndex) ? 1.0 : 0.0;
            double predictedValue = outputs[i];
            double residual = trueValue - predictedValue;

            // Apply the Huber loss formula for each class
            if (Math.abs(residual) <= DELTA) {
                // Quadratic loss for small residuals
                loss += 0.5 * residual * residual;
            } else {
                // Linear loss for large residuals, adjusted by the delta value
                loss += DELTA * (Math.abs(residual) - 0.5 * DELTA);
            }
        }
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


    // this will require summation and then division across multiple iterations over the back propagation
    public static double meanStandardLogarithmicError(double trueValue, double predictedValue){
        predictedValue = Math.max(predictedValue, 1e-15);
        trueValue = Math.max(trueValue, 1e-15);

        double logTrueValue = Math.log(trueValue + 1);
        double logPredictedValue = Math.log(predictedValue + 1);

        return Math.pow(logPredictedValue - logTrueValue, 2);
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
     * @param outputs Array of predicted probabilities for each class.
     * @param trueClass The actual class index.
     * @return The NLL value for the given observation.
     */
    private static double negativeLogLikelihood(int trueClass, double[] outputs) {
        double predictedProbability = outputs[trueClass];
        double epsilon = 1e-15; // a small number to prevent log(0)
        printGeneric("Predicted Probability = " + predictedProbability);
        predictedProbability = Math.max(outputs[trueClass], epsilon);
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
