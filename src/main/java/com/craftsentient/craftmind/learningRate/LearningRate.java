package com.craftsentient.craftmind.learningRate;


public class LearningRate {

    public static double epochDecay(double learningRate, double decayRate, int iteration) {
        return learningRate * (1 / (1 + decayRate * iteration));
    }

    public static double stepDecay(double learningRate, double decayRate, int step) {
        return learningRate * Math.pow(decayRate, step);
    }

    public static double exponentialDecay(double learningRate, double decayRate, int epoch) {
        return learningRate * Math.exp(-decayRate * epoch);
    }

    public static double AdagradAdaptiveDecay(double learningRate, double decayRate, double gradient, int iteration, double alpha) {
        return learningRate / Math.sqrt(gradient + alpha);
    }

    public static double rmsPropAdaptiveDecay(){
        return 0.0;
    }

    public static double adamAdaptiveDecay(){
        return 0.0;
    }
}
