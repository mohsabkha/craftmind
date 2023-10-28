package com.craftsentient.craftmind.utils;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MathUtils {

    public static Double dotProduct(Object a, Object b) {
        if (isDoubleList(a) && isDoubleList(b)) {
            ArrayList<Double> vec1 = (ArrayList<Double>) a;
            ArrayList<Double> vec2 = (ArrayList<Double>) b;
            if (vec1.size() != vec2.size())
                throw new IllegalArgumentException("Dimensions mismatch");
            AtomicReference<Double> sum = new AtomicReference<>((double) 0);
            IntStream.range(0, vec1.size()).parallel().forEachOrdered(i -> sum.updateAndGet(v -> (double) (v + vec1.get(i) * vec2.get(i))));
            return sum.getAcquire();
        } else if (isObjectList(a) && isObjectList(a)) {
            ArrayList<Object> tensor1 = (ArrayList<Object>) a;
            ArrayList<Object> tensor2 = (ArrayList<Object>) b;
            if (tensor1.size()!= tensor2.size())
                throw new IllegalArgumentException("Dimensions mismatch");
            AtomicReference<Double> sum = new AtomicReference<>((double) 0);
            IntStream.range(0, tensor1.size()).parallel().forEachOrdered( i -> sum.updateAndGet(v ->  v + dotProduct(tensor1.get(i), tensor2.get(i))));
            return sum.getAcquire();
        } else throw new IllegalArgumentException("Input must be arrays of the same dimension, and if they are, they are not arrayLists!");
    }

    public static double arrayDotProduct(double[]a, double[]b) {
        if(a.length != b.length) throw new IllegalArgumentException("Dimensions mismatch!");
        AtomicReference<Double> sum = new AtomicReference<>((double) 0);
        IntStream.range(0, a.length).parallel().forEachOrdered(i -> sum.updateAndGet(v -> (v + (a[i] * b[i]))));
        return sum.getAcquire();
    }

<<<<<<< HEAD
    public static double arrayDotProduct(ArrayList<Double>a, ArrayList<Double>b){
        if(a.size() != b.size()) throw new IllegalArgumentException("Dimensions mismatch!");
        AtomicReference<Double> sum = new AtomicReference<>((double) 0);
        IntStream.range(0, a.size()).parallel().forEachOrdered(i -> sum.updateAndGet(v -> (v + (a.get(i) * b.get(i)))));
        return sum.getAcquire();
    }


=======
>>>>>>> 1ed61536b58039b1964c30d9d2b649d4d7eaf800
    public static double[][] matrixDotProduct(double[][] input1, double[][] input2) {
        double[][] output = new double[input1.length][input2[0].length];
        for (int i = 0; i < output.length; i++)
            for (int j = 0; j < output[0].length; j++) {
                double value = 0;
                for (int k = 0; k < input1[0].length; k++)
                    value += input1[i][k] * input2[k][j];
                output[i][j] = value;
            }
        return output;
    }

    public static Object matrixDotProduct(Object a, Object b) {
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

    private static boolean bothDoubleArraysAreNotOfTheSameSize(Object a, Object b) {
        return ((ArrayList<Double>)a).size() != ((ArrayList<Double>)b).size();
    }

    private static boolean ElementOfFirstArrayIsDoubleArrayAndBothAreOfSameSize(Object a, Object b) {
        return isDoubleList(((ArrayList<?>)a).get(0)) && ( ((ArrayList<?>)((ArrayList<?>)a).get(0)).size() == ((ArrayList<Double>)b).size());
    }

    private static boolean ElementsOfBothArraysAreDoubleArraysAndBothAreOfProperSize(Object a, Object b) {
        return isDoubleList(((ArrayList<?>)a).get(0)) && isDoubleList(((ArrayList<?>)b).get(0)) && ( ((ArrayList<Double>)b).size() == ((ArrayList<?>)((ArrayList<?>)a).get(0)).size());
    }

    private static Double MutliplyVectors(ArrayList<Double> a, ArrayList<Double> b) {
        AtomicReference<Double> sum = new AtomicReference<>(0.0);
        IntStream.range(0, a.size()).parallel().forEachOrdered( i -> sum.updateAndGet(v -> v + a.get(i) * b.get(i)));
        return sum.get();
    }

    public static ArrayList<Double> MultipleMatrixAndVector(ArrayList<ArrayList<Double>> matrix, ArrayList<Double> vector) {
        ArrayList<Double> answer = new ArrayList<>();
        IntStream.range(0, matrix.size()).parallel().forEachOrdered( i -> answer.add(MutliplyVectors(matrix.get(i), vector)));
        return answer;
    }

    private static boolean isDoubleList(Object obj) {
        return obj instanceof ArrayList && !((ArrayList<?>) obj).isEmpty() && ((ArrayList<?>) obj).get(0) instanceof Double;
    }

    private static boolean isObjectList(Object obj) {
        return obj instanceof ArrayList && !((ArrayList<?>) obj).isEmpty() && ((ArrayList<?>) obj).get(0) instanceof Object;
    }

    private static Object returnInnerDoubleArrays(Object obj) {
        if(isObjectList(obj) && isDoubleList(((ArrayList<?>) obj).get(0))) return obj;
        else if (isDoubleList(obj)) return obj;
        else return returnInnerDoubleArrays(obj);
    }

    private static ArrayList<Double> returnDoubleArray(ArrayList<ArrayList<Double>> objectArray, int index) {
        return objectArray.get(index);
    }

    private static ArrayList<ArrayList<Double>> returnArrayOfDoubleArrays(ArrayList<ArrayList<Double>> matrix1, ArrayList<ArrayList<Double>> matrix2) {
        ArrayList<ArrayList<Double>> answer = new ArrayList<>();
        AtomicReference<ArrayList<Double>> temp = new AtomicReference<>();
        IntStream.range(0, matrix2.get(0).size()).parallel().forEachOrdered( i -> {
            temp.set(new ArrayList<>());
            IntStream.range(0, matrix2.size()).parallel().forEachOrdered( j -> { temp.get().add(matrix2.get(j).get(i)); });
            answer.add(MultipleMatrixAndVector(matrix1, temp.get()));
        });
       return transposeMatrix(answer);
    }

    public static ArrayList<ArrayList<Double>> transposeMatrix(ArrayList<ArrayList<Double>> matrix) {
        ArrayList<ArrayList<Double>> finalAnswer = new ArrayList<>();
        AtomicReference<ArrayList<Double>> temp = new AtomicReference<>();
        IntStream.range(0, matrix.get(0).size()).parallel().forEachOrdered(i -> {
            temp.set(new ArrayList<>());
            IntStream.range(0, matrix.size()).parallel().forEachOrdered(j -> { temp.get().add(matrix.get(i).get(j)); });
            finalAnswer.add(temp.get());
        });
        return finalAnswer;
    }

    public static double[][] transposeMatrix(double[][] matrix) {
        double[][] finalAnswer = new double[matrix[0].length][matrix.length];
        IntStream.range(0, matrix[0].length).parallel().forEachOrdered(i -> {
            double[] temp = new double[matrix.length];
            IntStream.range(0, matrix.length).parallel().forEachOrdered(j -> {
                temp[j] = matrix[j][i];
            });
            finalAnswer[i] = (temp);
        });
        return finalAnswer;
    }

    public static ArrayList<Double> addVectors(ArrayList<Double> a, ArrayList<Double> b) throws ArithmeticException {
        if(a.size() != b.size()) { throw new ArithmeticException("vectors should be of same size"); }
        if(a.size() == 0) throw new ArithmeticException("vector sizes should not be zero");
        ArrayList<Double> vectorSum = new ArrayList<>(a.size());
        IntStream.range(0, a.size()).parallel().forEachOrdered( i -> { vectorSum.add(a.get(i) + b.get(i)); });
        return vectorSum;
    }

    public static double[] addArrays(double[] a, double[] b) {
        if(a.length != b.length) { throw new ArithmeticException("vectors should be of same size"); }
        if(a.length == 0) throw new ArithmeticException("vector sizes should not be zero");
        double[] vectorSum = new double[a.length];
        IntStream.range(0, a.length).parallel().forEachOrdered( i -> { vectorSum[i] = (a[i] + b[i]); });
        return vectorSum;
    }

    public static double[][] add(double[][] a, double[] b) {
        double[][] output = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < a[0].length; j++)
                output[i][j] = a[i][j] + b[j];
        return output;
    }

    public static double[] addToDoubleArray(double[] array, double value) {
        double[] temp;
        if(array.length == 0) {
            temp = new double[1];
            temp[0] = value;
        } else {
            temp = new double[array.length + 1];
            for(int i = 0; i < array.length; i++) temp[i] = array[i];
            temp[array.length] = value;
        }
        return temp;
    }

    public static double[][] addToDoubleArray(double[][] array, double[] values) {
        double[][] temp;
        if(array.length == 0){
            temp = new double[1][values.length];
            temp[0] = values;
        } else {
            temp = new double[array.length+1][array[0].length];
            for(int i = 0; i < array.length; i++) temp[i] = array[i];
            temp[array.length] = values;
        }
        return temp;
    }

    public static double[] fullMultiplication(double[] a, double[] b) {
        return IntStream.range(0, a.length).parallel().mapToDouble(i -> {
            return IntStream.range(0, b.length).mapToDouble(j -> a[i] * b[j]).sum();
        }).toArray();
    }

    public static ArrayList<Double> arrayToArrayList(double[] array) {
        return Arrays.stream(array).boxed().collect(Collectors.toCollection(ArrayList::new));
    }

    private static double exp(double value) {
        return Math.exp(value);
    }

    private static double[] exp(double[] values) {
        IntStream.range(0, values.length).parallel().forEachOrdered(i -> values[i] = exp(values[i]));
        return values;
    }

    public static double mean(double[] values){
        double meanLoss = 0;
        for(int i = 0; i < values.length; i++)
            meanLoss += values[i];
        return meanLoss/3.0;
    }

    public static double[] mean(double[][] values){
        double[] meanLoss = new double[values.length];
        for(int i = 0; i < values.length; i++)
            meanLoss[i] = mean(values[i]);
        return meanLoss;
    }

    public static double max(double[] values){
        double max = 0;
        for(int i = 0; i < values.length; i++)
            if(max < values[i]) max = values[i];
        return max;
    }

    public static double[] indexAndMax(double[] values){
        double[] indexAndMax = new double[2];
        for(int i = 0; i < values.length; i++)
            if(indexAndMax[1] < values[i]) {
                indexAndMax[0] = i;
                indexAndMax[1] = values[i];
            };
        return indexAndMax;
    }

    public static int[] getHotOneVecIndexValues(int[][] hotOneVec){
        int[] indices = new int[hotOneVec.length];
        IntStream.range(0, indices.length).parallel().forEachOrdered( i -> {
            IntStream.range(0, hotOneVec[i].length).forEach( j -> {
                if(hotOneVec[i][j] == 1) indices[i] = j;
            });
        });
        return indices;
    }

    public static double accuracySum(double[] trueValues, int[] predictedValues){
        if(trueValues.length != predictedValues.length){throw new IllegalArgumentException("Number of true values is not equal to number of predicted values");}
        AtomicReference<Double> totalCorrect = new AtomicReference<>((double)0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> {
            if(trueValues[i] == predictedValues[i]) totalCorrect.getAndUpdate(v -> v + 1);
        });
        return totalCorrect.get();
    }

    public static double accuracySum(double[] trueValues, double[] predictedValues){
        if(trueValues.length != predictedValues.length){throw new IllegalArgumentException("Number of true values is not equal to number of predicted values");}
        AtomicReference<Double> totalCorrect = new AtomicReference<>((double)0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> {
            if(trueValues[i] == predictedValues[i]) totalCorrect.getAndUpdate(v -> v + 1);
        });
        return totalCorrect.get();
    }

    public static double accuracySum(int[] trueValues, int[] predictedValues){
        if(trueValues.length != predictedValues.length){throw new IllegalArgumentException("Number of true values is not equal to number of predicted values");}
        AtomicReference<Double> totalCorrect = new AtomicReference<>((double)0);
        IntStream.range(0, trueValues.length).parallel().forEachOrdered(i -> {
            if(predictedValues[i] == trueValues[i]) totalCorrect.getAndUpdate(v -> v + 1);
        });
        return totalCorrect.get();
    }

    public static double accuracySum(int[][] trueHotValues, int[] predictedValues){
        if(trueHotValues.length != predictedValues.length){throw new IllegalArgumentException("Number of true values is not equal to number of predicted values");}
        AtomicReference<Double> totalCorrect = new AtomicReference<>((double)0);
        int[] trueValues = getHotOneVecIndexValues(trueHotValues);
        IntStream.range(0, trueHotValues.length).parallel().forEachOrdered(i -> {
            if(predictedValues[i] == trueValues[i]) totalCorrect.getAndUpdate(v -> v + 1);
        });
        return totalCorrect.get();
    }

    public static double accuracy(double[] trueValues, double[] predictedValues){
        double sum = accuracySum(trueValues, predictedValues);
        return sum/trueValues.length;
    }

    public static double accuracy(double[] trueValues, int[] predictedValues){
        double sum = accuracySum(trueValues, predictedValues);
        return sum/trueValues.length;
    }

    public static double accuracy(int[] trueValues, int[] predictedValues){
        double sum = accuracySum(trueValues, predictedValues);
        return sum/trueValues.length;
    }

    public static double accuracy(int[][] trueHotValues, int[] predictedValues){
        double sum = accuracySum(trueHotValues, predictedValues);
        return sum/trueHotValues.length;
    }

    public static void print(double[][] matrix) {
        System.out.println("Matrix (" + matrix.length +" x " + matrix[0].length + ")");
        for (int i = 0; i < matrix.length; i++) {
            System.out.print("Row " + (i+1) + " : [");
            for (int j = 0; j < matrix[0].length; j++)
                if (j != matrix[0].length - 1)
                    System.out.print(matrix[i][j] + ", ");
                else System.out.print(matrix[i][j] + "");
            System.out.println("]");
        }
    }

    public static void print(double[][] matrix, String name) {
        System.out.println(name + " {Matrix (" + matrix.length +" x " + matrix[0].length + ")}");
        for (int i = 0; i < matrix.length; i++) {
            System.out.print("Row " + (i+1) + " : [");
            for (int j = 0; j < matrix[0].length; j++) {
                if (j != matrix[0].length - 1)
                    System.out.print(matrix[i][j] + ", ");
                else System.out.print(matrix[i][j] + "");
            }
            System.out.println("]");
        }
    }
}
