package com.craftsentient.craftmind.utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.stream.IntStream;

public class FileUtils {

    public static Object readTextFile(String filePathForInputs, String delimiter){
        ArrayList<double[]> inputs = new ArrayList<>();
        try {
            FileReader fileReader = new FileReader(filePathForInputs);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line;
            while((line = bufferedReader.readLine()) != null){
                ArrayList<Double> input = new ArrayList<>();
                String[] cells = line.split(delimiter);
                for(String cell : cells){
                    input.add(Double.parseDouble(cell));
                }
                inputs.add(input.stream().mapToDouble(Double::doubleValue).toArray());
            }
            bufferedReader.close();
            double[][] finalInputs = new double[inputs.size()][];
            IntStream.range(0, inputs.size()).parallel().forEachOrdered( i -> finalInputs[i] = inputs.get(i));
            return finalInputs;
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static double[][] readCsvFile(String filePathForInputs){
        ArrayList<double[]> inputs = new ArrayList<>();
        try {
            FileReader fileReader = new FileReader(filePathForInputs);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line;
            while((line = bufferedReader.readLine()) != null){
                ArrayList<Double> input = new ArrayList<>();
                String[] cells = line.split(",");
                for(String cell : cells){
                    input.add(Double.parseDouble(cell));
                }
                inputs.add(input.stream().mapToDouble(Double::doubleValue).toArray());
            }
            bufferedReader.close();
            double[][] finalInputs = new double[inputs.size()][];
            IntStream.range(0, inputs.size()).parallel().forEachOrdered( i -> finalInputs[i] = inputs.get(i));
            return finalInputs;
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

//    public static double[] readCsvFile(String filePathForTrueValues){
//        ArrayList<double[]> inputs = new ArrayList<>();
//        try {
//            FileReader fileReader = new FileReader(filePathForTrueValues);
//            BufferedReader bufferedReader = new BufferedReader(fileReader);
//            String line;
//            while((line = bufferedReader.readLine()) != null){
//                ArrayList<Double> input = new ArrayList<>();
//                String[] cells = line.split(",");
//                for(String cell : cells){
//                    input.add(Double.parseDouble(cell));
//                }
//                inputs.add(input.stream().mapToDouble(Double::doubleValue).toArray());
//            }
//            bufferedReader.close();
//            double[][] finalInputs = new double[inputs.size()];
//            IntStream.range(0, inputs.size()).parallel().forEachOrdered( i -> finalInputs[i] = inputs.get(i));
//            return finalInputs;
//        } catch (FileNotFoundException e) {
//            throw new RuntimeException(e);
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
//    }

}
