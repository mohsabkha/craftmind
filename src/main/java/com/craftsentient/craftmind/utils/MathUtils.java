package com.craftsentient.craftmind.utils;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

public class MathUtils {

    public static Double dotProduct(Object a, Object b) {
        if (isDoubleList(a) && isDoubleList(b)) {
            ArrayList<Double> vec1 = (ArrayList<Double>) a;
            ArrayList<Double> vec2 = (ArrayList<Double>) b;

            if (vec1.size() != vec2.size()) {
                throw new IllegalArgumentException("Dimensions mismatch");
            }

            AtomicReference<Double> sum = new AtomicReference<>((double) 0);
            IntStream.range(0, vec1.size()).parallel().forEach(i -> {
                sum.updateAndGet(v -> (double) (v + vec1.get(i) * vec2.get(i)));
            });
            return sum.getAcquire();
        } else if (isObjectList(a) && isObjectList(a)) {
            ArrayList<Object> tensor1 = (ArrayList<Object>) a;
            ArrayList<Object> tensor2 = (ArrayList<Object>) b;

            if (tensor1.size()!= tensor2.size()) {
                throw new IllegalArgumentException("Dimensions mismatch");
            }

            AtomicReference<Double> sum = new AtomicReference<>((double) 0);
            IntStream.range(0, tensor1.size()).parallel().forEach( i -> {
                sum.updateAndGet(v ->  v + dotProduct(tensor1.get(i), tensor2.get(i)));
            });
            return sum.getAcquire();
        } else {
            throw new IllegalArgumentException("Input must be arrays of the same dimension");
        }
    }

//    public static void matrixDotProduct(Object a, Object b){
//        // check to see if the first objects columns match second objects rows
//        // --> a.get(0).size() == b.size()
//            // if true then create a matrix that has the size of the columnA by rowB
//            // then for each position, find the dot product for ColumnA and RowB and fill in that position
//
//        if(isDoubleList(a) && isDoubleList(b)){
//            if( ((ArrayList<Double>)a).size() != ((ArrayList<Double>)a).size() ){
//                throw new ArithmeticException("a and b must be of equal size");
//            }
//        } else if( isObjectList(a) && isDoubleList(b) ){
//            if( isDoubleList(((ArrayList<?>)a).get(0)) && ( a.get(0).size() == ((ArrayList<Double>)b).size()) ){
//
//            }
//        } else if(isDoubleList(a) && isObjectList(b)){
//
//        } else if(isObjectList(a) && isObjectList(b)){
//
//        }
//
//    }

    private static boolean isDoubleList(Object obj){
        return obj instanceof ArrayList &&
            !((ArrayList<?>) obj).isEmpty() &&
            ((ArrayList<?>) obj).get(0) instanceof Double;
    }

    private static boolean isObjectList(Object obj){
        return obj instanceof ArrayList &&
                !((ArrayList<?>) obj).isEmpty() &&
                ((ArrayList<?>) obj).get(0) instanceof Object;
    }

    // returns the matrix that holds the lists of doubles
    public static Object returnInnerDoubleArrays(Object obj){
        if(isObjectList(obj) && isDoubleList(((ArrayList<?>) obj).get(0))){
            return obj;
        } else if (isDoubleList(obj)){
            return obj;
        } else {
            return returnInnerDoubleArrays(obj);
        }
    }

    public static ArrayList<Double> addVectors(ArrayList<Double> a, ArrayList<Double> b) throws ArithmeticException {
        if(a.size() != b.size()) { throw new ArithmeticException("vectors should be of same size"); }
        if(a.size() == 0) throw new ArithmeticException("vector sizes should not be zero");
        ArrayList<Double> vectorSum = new ArrayList<>(a.size());
        IntStream.range(0, a.size()).parallel().forEach( i -> { vectorSum.add(a.get(i) + b.get(i)); });
        return vectorSum;
    }
}
