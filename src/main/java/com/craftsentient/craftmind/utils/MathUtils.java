package com.craftsentient.craftmind.utils;


import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MathUtils {

    public static Object dotProduct(Object a, Object b) {
        if(isDoubleList(a) && isDoubleList(b)){
            if (((ArrayList<Double>)a).size() != ((ArrayList<Double>)b).size()) {
                throw new IllegalArgumentException("Lists must be of the same length.");
            }
            return IntStream.range(0, ((ArrayList<Double>)a).size())
                    .parallel()
                    .mapToDouble(i -> ((ArrayList<Double>)a).get(i) * ((ArrayList<Double>)b).get(i))
                    .sum();

        } else if (a instanceof ArrayList && b instanceof ArrayList) {
            if (((ArrayList<Double>)a).size() != ((ArrayList<Double>)b).size()) {
                throw new IllegalArgumentException("Lists must be of the same length.");
            }
            return IntStream.range(0, ((ArrayList<Double>)a).size())
                    .parallel()
                    .mapToObj(i -> dotProduct(((ArrayList<Double>)a).get(i), ((ArrayList<Double>)b).get(i)))
                    .collect(Collectors.toCollection(ArrayList::new));
        }
        else {
            throw new IllegalArgumentException("Inputs must be ArrayLists or nested ArrayLists of equal structure.");
        }
    }

    private static boolean isDoubleList(Object matrix) {
        return
            matrix instanceof ArrayList &&
            !((ArrayList<?>) matrix).isEmpty() &&
            ((ArrayList<?>) matrix).get(0) instanceof Double;
    }

    public static ArrayList<Double> addVectors(ArrayList<Double> a, ArrayList<Double> b) throws ArithmeticException {
        if(a.size() != b.size()) { throw new ArithmeticException("vectors should be of same size"); }
        if(a.size() == 0) throw new ArithmeticException("vector sizes should not be zero");
        ArrayList<Double> vectorSum = new ArrayList<>(a.size());
        for(int i = 0; i < a.size(); i++) { vectorSum.add(a.get(i) + b.get(i)); }
        return vectorSum;
    }
}
