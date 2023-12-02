package com.craftsentient.craftmind.learningRate;

public class LearningRateImpl {
    public static double SgdWithMomentum(double learningRate, double decayRate, int epoch){
        return learningRate;
    }

    public static double epochDecay(double learningRate, double decayRate, int epoch) {
        if((1 + decayRate * epoch) != 0) {
            return learningRate * (1 / (1 + decayRate * epoch));
        } else {
            return learningRate * (1 / (1 + decayRate * epoch) + 1e-15);
        }
    }

    public static double stepDecay(double learningRate, double decayRate, int step) {
        return learningRate * Math.pow(decayRate, step);
    }

    public static double exponentialDecay(double learningRate, double decayRate, int epoch) {
        return learningRate * Math.exp(-decayRate * epoch);
    }

    public static double AdagradAdaptiveDecay(double learningRate, double gradient, double alpha) {
        if(Math.sqrt(gradient + alpha) != 0) {
            return learningRate / Math.sqrt(gradient + alpha);
        } else {
            return learningRate / Math.sqrt(gradient + alpha) + 1e-15;
        }

    }

    public static double rmsPropAdaptiveDecay(){
        return 0.0;
    }

    public static double adamAdaptiveDecay(){
        return 0.0;
    }
}
