package com.craftsentient.craftmind.learningRate;


import static com.craftsentient.craftmind.learningRate.LearningRateImpl.*;

public class LearningRate {
    public static double updateLearningRate(DEFAULT_LEARNING_RATE decayFunction, double learningRate, double decayRate, int iteration){
        switch (decayFunction){
            case EPOCH: {
                return epochDecay(learningRate, decayRate, iteration);
            }
            case STEP: {
                return stepDecay(learningRate, decayRate, iteration);
            }
            case EXPONENTIAL: {
                // return exponentialDecay();
            }
            case ADAGRAD: {
                return adamAdaptiveDecay();
            }
            case RMSPROP: {
                return rmsPropAdaptiveDecay();
            }
            case ADAM: {
                return adamAdaptiveDecay();
            }
            default: return 0.01;
        }
    }
}
