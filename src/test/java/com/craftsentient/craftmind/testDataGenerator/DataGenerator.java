package com.craftsentient.craftmind.testDataGenerator;

import java.util.Random;

public class DataGenerator {

    public static Object[] createData(int samples, int classes) {
        double[][] X = new double[samples * classes][2];
        int[] y = new int[samples * classes];
        Random rand = new Random();

        for (int classNumber = 0; classNumber < classes; classNumber++) {
            for (int i = 0; i < samples; i++) {
                int ix = i + samples * classNumber;
                double r = (double) i / (samples - 1);
                double t = (double) (classNumber * 4) + ((double) i / (samples - 1)) * 4 + rand.nextGaussian() * 0.2;
                X[ix][0] = r * Math.sin(t * 2.5);
                X[ix][1] = r * Math.cos(t * 2.5);
                y[ix] = classNumber;
            }
        }

        return new Object[]{X, y};
    }

}
