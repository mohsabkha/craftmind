package com.craftsentient.craftmind.utils;


import java.util.ArrayList;

public class MathUtils {

    public static double dotProduct(ArrayList<Double> a, ArrayList<Double> b) {
        if(a.size() != b.size()) { throw new ArithmeticException(); }
        double output = 0.0;
        for(int i = 0; i < a.size(); i++){
            output += (a.get(i) * b.get(i));
        }
        return output;
    }

    public static ArrayList<Double> addVectors(ArrayList<Double> a, ArrayList<Double> b) throws ArithmeticException {
        if(a.size() != b.size()) { throw new ArithmeticException("vectors should be of same size"); }
        if(a.size() == 0) throw new ArithmeticException("vector sizes should not be zero");
        ArrayList<Double> vectorSum = new ArrayList<>(a.size());
        for(int i = 0; i < a.size(); i++) { vectorSum.add(a.get(i) + b.get(i)); }
        return vectorSum;
    }
}
