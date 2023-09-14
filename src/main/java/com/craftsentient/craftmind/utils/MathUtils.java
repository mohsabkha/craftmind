package com.craftsentient.craftmind.utils;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

public class MathUtils {

    public static Double dotProduct(Object a, Object b) {
        if (isDoubleList(a) && isDoubleList(b)) {
            ArrayList<Double> vec1 = (ArrayList<Double>) a;
            ArrayList<Double> vec2 = (ArrayList<Double>) b;
            if (vec1.size() != vec2.size())
                throw new IllegalArgumentException("Dimensions mismatch");
            AtomicReference<Double> sum = new AtomicReference<>((double) 0);
            IntStream.range(0, vec1.size()).forEach(i -> sum.updateAndGet(v -> (double) (v + vec1.get(i) * vec2.get(i))));
            return sum.getAcquire();

        } else if (isObjectList(a) && isObjectList(a)) {
            ArrayList<Object> tensor1 = (ArrayList<Object>) a;
            ArrayList<Object> tensor2 = (ArrayList<Object>) b;
            if (tensor1.size()!= tensor2.size())
                throw new IllegalArgumentException("Dimensions mismatch");
            AtomicReference<Double> sum = new AtomicReference<>((double) 0);
            IntStream.range(0, tensor1.size()).forEach( i -> sum.updateAndGet(v ->  v + dotProduct(tensor1.get(i), tensor2.get(i))));
            return sum.getAcquire();
        } else throw new IllegalArgumentException("Input must be arrays of the same dimension");
    }

    public static Object matrixDotProduct(Object a, Object b){
        if(isDoubleList(a) && isDoubleList(b)){
            if(bothDoubleArraysAreNotOfTheSameSize(a,b))
                throw new ArithmeticException("ERR1: a and b must be of equal size");
            return MutliplyVectors((ArrayList<Double>) a, (ArrayList<Double>) b);
        } else if( isObjectList(a) && isDoubleList(b) ){
            if(ElementOfFirstArrayIsDoubleArrayAndBothAreOfSameSize(a,b)){
                ArrayList<ArrayList<Double>> matrix = (ArrayList<ArrayList<Double>>) a;
                ArrayList<Double> vector = (ArrayList<Double>) b;
                return MultipleMatrixAndVector(matrix, vector);
            }
            else throw new ArithmeticException("ERR2: a and b must be of equal size");
        } else if(isObjectList(b) && isDoubleList(a)){
            if(ElementOfFirstArrayIsDoubleArrayAndBothAreOfSameSize(b,a)){
                ArrayList<ArrayList<Double>> matrix = (ArrayList<ArrayList<Double>>) b;
                ArrayList<Double> vector = (ArrayList<Double>) a;
                return MultipleMatrixAndVector(matrix, vector);
            }
            else throw new ArithmeticException("ERR3: a and b must be of equal size");
        } else if(isObjectList(a) && isObjectList(b)){
            if(ElementsOfBothArraysAreDoubleArraysAndBothAreOfProperSize(a,b)){
                ArrayList<ArrayList<Double>> matrix1 = (ArrayList<ArrayList<Double>>) a;
                ArrayList<ArrayList<Double>> matrix2 = (ArrayList<ArrayList<Double>>) b;
                return returnArrayOfDoubleArrays(matrix1, matrix2);
            }
            else throw new ArithmeticException("ERR4: a and b must be of equal size");
        }
        else throw new ArithmeticException("ERR5: a and b must be of equal size");
    }

    private static boolean bothDoubleArraysAreNotOfTheSameSize(Object a, Object b){
        return ((ArrayList<Double>)a).size() != ((ArrayList<Double>)b).size();
    }

    private static boolean ElementOfFirstArrayIsDoubleArrayAndBothAreOfSameSize(Object a, Object b){
        return isDoubleList(((ArrayList<?>)a).get(0)) && ( ((ArrayList<?>)((ArrayList<?>)a).get(0)).size() == ((ArrayList<Double>)b).size());
    }

    private static boolean ElementsOfBothArraysAreDoubleArraysAndBothAreOfProperSize(Object a, Object b){
        return
                isDoubleList(((ArrayList<?>)a).get(0))
                && isDoubleList(((ArrayList<?>)b).get(0))
                && ( ((ArrayList<Double>)b).size() == ((ArrayList<?>)((ArrayList<?>)a).get(0)).size());
    }

    private static Double MutliplyVectors(ArrayList<Double> a, ArrayList<Double> b){
        AtomicReference<Double> sum = new AtomicReference<>(0.0);
        IntStream.range(0, a.size()).forEach( i -> sum.updateAndGet(v -> v + a.get(i) * b.get(i)));
        return sum.get();
    }

    public static ArrayList<Double> MultipleMatrixAndVector(ArrayList<ArrayList<Double>> matrix, ArrayList<Double> vector){
        ArrayList<Double> answer = new ArrayList<>();
        IntStream.range(0, matrix.size()).forEach( i -> answer.add(MutliplyVectors(matrix.get(i), vector)));
        return answer;
    }

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
    private static Object returnInnerDoubleArrays(Object obj){
        if(isObjectList(obj) && isDoubleList(((ArrayList<?>) obj).get(0))){
            return obj;
        } else if (isDoubleList(obj)){
            return obj;
        } else {
            return returnInnerDoubleArrays(obj);
        }
    }

    private static ArrayList<Double> returnDoubleArray(ArrayList<ArrayList<Double>> objectArray, int index){return objectArray.get(index);}

    private static ArrayList<ArrayList<Double>> returnArrayOfDoubleArrays(ArrayList<ArrayList<Double>> matrix1, ArrayList<ArrayList<Double>> matrix2){
        ArrayList<ArrayList<Double>> answer = new ArrayList<>();
        AtomicReference<ArrayList<Double>> temp = new AtomicReference<>();
        IntStream.range(0, matrix2.get(0).size()).forEach( i -> {
            temp.set(new ArrayList<>());
            IntStream.range(0, matrix2.size()).forEach( j -> {
                temp.get().add(matrix2.get(j).get(i));
            });
            answer.add(MultipleMatrixAndVector(matrix1, temp.get()));
        });
       return transposeMatrix(answer);
    }

    public static ArrayList<ArrayList<Double>> transposeMatrix(ArrayList<ArrayList<Double>> matrix){
        ArrayList<ArrayList<Double>> finalAnswer = new ArrayList<>();
        ArrayList<Double> temp2;
        for(int x = 0; x < matrix.get(0).size(); x++){
            temp2 = new ArrayList<>();
            for(int y = 0; y < matrix.size(); y++){
                temp2.add(matrix.get(y).get(x));
            }
            finalAnswer.add(temp2);
        }
        return finalAnswer;
    }

    public static ArrayList<Double> addVectors(ArrayList<Double> a, ArrayList<Double> b) throws ArithmeticException {
        if(a.size() != b.size()) { throw new ArithmeticException("vectors should be of same size"); }
        if(a.size() == 0) throw new ArithmeticException("vector sizes should not be zero");
        ArrayList<Double> vectorSum = new ArrayList<>(a.size());
        IntStream.range(0, a.size()).forEach( i -> { vectorSum.add(a.get(i) + b.get(i)); });
        return vectorSum;
    }
}
