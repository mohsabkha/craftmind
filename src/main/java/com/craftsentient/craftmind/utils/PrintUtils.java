package com.craftsentient.craftmind.utils;

import com.craftsentient.craftmind.layer.DenseLayer;
import com.craftsentient.craftmind.neuralNetwork.BaseNeuralNetwork;
import com.craftsentient.craftmind.neuron.Neuron;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class PrintUtils {
    private static String BLACK = "\033[30m";
    private static String RED = "\033[31m";
    private static String GREEN = "\033[32m";
    private static String YELLOW = "\033[33m";
    private static String BLUE = "\033[34m";
    private static String PURPLE = "\033[35m";
    private static String CYAN = "\033[36m";
    private static String WHITE = "\033[37m";
    private static String BLACK_BACKGROUND = "\033[40m";
    private static String RED_BACKGROUND  = "\033[41m";
    private static String GREEN_BACKGROUND  = "\033[42m";
    private static String YELLOW_BACKGROUND  = "\033[43m";
    private static String BLUE_BACKGROUND  = "\033[44m";
    private static String PURPLE_BACKGROUND  = "\033[45m";
    private static String CYAN_BACKGROUND  = "\033[46m";
    private static String WHITE_BACKGROUND  = "\033[47m";
    private static String BOLD = "\033[1m";
    private static String RESET = "\033[0m";
    public static String black(String str){
        return BLACK + str + RESET;
    }
    public static String red(String str){
        return RED + str + RESET;
    }
    public static String green(String str){
        return GREEN + str + RESET;
    }
    public static String yellow(String str){
        return YELLOW + str + RESET;
    }
    public static String blue(String str){
        return BLUE + str + RESET;
    }
    public static String purple(String str){
        return PURPLE + str + RESET;
    }
    public static String cyan(String str){
        return CYAN + str + RESET;
    }
    public static String white(String str){
        return WHITE + str + RESET;
    }
    public static String backgroundBlack(String str){
        return BLACK_BACKGROUND + str + RESET;
    }
    public static String backgroundRed(String str){
        return RED_BACKGROUND + str + RESET;
    }
    public static String backgroundGreen(String str){
        return GREEN_BACKGROUND + str + RESET;
    }
    public static String backgroundYellow(String str){
        return YELLOW_BACKGROUND + str + RESET;
    }
    public static String backgroundBlue(String str){
        return BLUE_BACKGROUND + str + RESET;
    }
    public static String backgroundPurple(String str){
        return PURPLE_BACKGROUND + str + RESET;
    }
    public static String backgroundCyan(String str){
        return CYAN_BACKGROUND + str + RESET;
    }
    public static String backgroundWhite(String str){
        return WHITE_BACKGROUND + str + RESET;
    }
    public static String bold(String str){
        return BOLD + str + RESET;
    }

    // generic
    public static void printTitle(String data){
        System.out.println(backgroundGreen(black(bold(":::: " + data + " ::::"))));
    }
    public static void printSubTitle(String data) {
        System.out.println(backgroundBlack(green(bold(":::: " + data + " ::::"))));
    }

    // positive
    public static void printPositive(String data){
        System.out.println(green(bold("[SUCCESS] ") + data));
    }
    public static void printPositive(double data){
        System.out.println(green(bold("[SUCCESS] ") + data));
    }
    public static void printPositive(int data){
        System.out.println(green(bold("[SUCCESS] ") + data));
    }
    public static void printPositive(float data){
        System.out.println(green(bold("[SUCCESS] ") + data));
    }
    public static void printPositive(short data){
        System.out.println(green(bold("[SUCCESS] ") + data));
    }
    public static void printPositive(long data){
        System.out.println(green(bold("[SUCCESS] ") + data));
    }
    public static void printPositive(double[] vec, String label){
        System.out.println(bold(green(":::: " + label + " ::::")));
        System.out.print(bold(green("[SUCCESS] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(green("]")));
            else if(i == 0) System.out.print(bold(green("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printPositive(int[] vec, String label){
        System.out.println(bold(green(":::: " + label + " ::::")));
        System.out.print(bold(green("[SUCCESS] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(green("]")));
            else if(i == 0) System.out.print(bold(green("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printPositive(String[] vec, String label){
        System.out.println(bold(green(":::: " + label + " ::::")));
        System.out.print(bold(green("[SUCCESS] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(green("]")));
            else if(i == 0) System.out.print(bold(green("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printPositive(ArrayList<Object> vec, String label){
        System.out.println(bold(green(":::: " + label + " ::::")));
        System.out.print(bold(green("[SUCCESS] ")));
        IntStream.range(0, vec.size()).forEach(i-> {
            if(i == vec.size()-1) System.out.print(vec.get(i) + bold(green("]")));
            else if(i == 0) System.out.print(bold(green("[")) + vec.get(i) + ", ");
            else System.out.print(String.valueOf(vec.get(i)) + ", ");
        });
        System.out.print("\n");
    }

    // positive
    public static void warning(String data){
        System.out.println(red(bold("[WARNING] ") + data));
    }
    public static void warning(double data){
        System.out.println(red(bold("[WARNING] ") + data));
    }
    public static void warning(int data){
        System.out.println(red(bold("[WARNING] ") + data));
    }
    public static void warning(float data){
        System.out.println(red(bold("[WARNING] ") + data));
    }
    public static void warning(short data){
        System.out.println(red(bold("[WARNING] ") + data));
    }
    public static void warning(long data){
        System.out.println(red(bold("[WARNING] ") + data));
    }
    public static void warning(double[] vec, String label){
        System.out.println(bold(red(":::: " + label + " ::::")));
        System.out.print(bold(red("[WARNING] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void warning(int[] vec, String label){
        System.out.println(bold(red(":::: " + label + " ::::")));
        System.out.print(bold(red("[WARNING] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void warning(String[] vec, String label){
        System.out.println(bold(red(":::: " + label + " ::::")));
        System.out.print(bold(red("[WARNING] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void warning(ArrayList<Object> vec, String label){
        System.out.println(bold(red(":::: " + label + " ::::")));
        System.out.print(bold(green("[WARNING] ")));
        IntStream.range(0, vec.size()).forEach(i-> {
            if(i == vec.size()-1) System.out.print(vec.get(i) + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec.get(i) + ", ");
            else System.out.print(String.valueOf(vec.get(i)) + ", ");
        });
        System.out.print("\n");
    }

    // no label
    public static void print(String data) { System.out.println(data); }
    public static void print(double data) { System.out.println(data); }
    public static void print(int data){
        System.out.println(data);
    }
    public static void print(long data){
        System.out.println(data);
    }
    public static void print(float data){
        System.out.println(data);
    }
    public static void print(short data){
        System.out.println(data);
    }
    public static void print(String info, double data){System.out.println(bold(info) + " " + data);}
    public static void print(String info, int data){System.out.println(bold(info) + " " + data);}
    public static void print(String info, long data){System.out.println(bold(info) + " " + data);}
    public static void print(String info, String data){System.out.println(bold(info) + " " + data);}

    public static void print(boolean[] vec) {
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(byte[] vec) {
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(double[] vec) {
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(float[] vec) {
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(int[] vec) {
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(long[] vec) {
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(short[] vec) {
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(BigDecimal[] vec) {
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(BigInteger[] vec) {
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String[] vec) {
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(vec[i] + ", ");
        });
        System.out.print("\n");
    }

    public static void print(String info, boolean[] vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, byte[] vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, double[] vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, float[] vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, int[] vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, long[] vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, short[] vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, BigDecimal[] vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, BigInteger[] vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, String[] vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(vec[i] + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, ArrayList<Object> vec) {
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.size()).forEach(i-> {
            if(i == vec.size()-1) System.out.print(vec.get(i) + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec.get(i) + ", ");
            else System.out.print(String.valueOf(vec.get(i)) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, Collection<Object> vec) {
        System.out.print(bold(info + " "));
        AtomicInteger counter = new AtomicInteger();
        vec.forEach(i-> {
            if(counter.get() == vec.size()-1) System.out.print( i + bold(blue("]")));
            else if(counter.get() == 0) System.out.print(blue(bold("[")) + i + ", ");
            else System.out.print(String.valueOf(i) + ", ");
            counter.getAndIncrement();
        });
        System.out.print("\n");
    }


    public static void printNeurons(String label, Neuron neuron) {
        System.out.println(bold(green("Neuron " + label + " ")));
        print("weights:", neuron.getWeights());
        print("bias: ", neuron.getBias());
    }

    public static void printLayer(String label, DenseLayer layer) {
        System.out.println(bold(blue("Layer " + label + " ")));
        print("Number of Neurons:", layer.getNeuronList().size());
        for(int i = 0; i < layer.getNeuronList().size(); i++){
            printNeurons(i + "", layer.getNeuronList().get(i));
        }
        System.out.println(bold(green("Overall Layer " + label + " Details")));
        print("Inputs: ", layer.getInputs());
        print("Outputs: ", layer.getLayerOutputs());
    }

    public static void printLayers(String label, BaseNeuralNetwork layers) {
        System.out.println(bold(green(":::: " + label + " NETWORK ::::")));
        for(int i = 1; i <= layers.getLayerList().size(); i++){
            printLayer("" + i, layers.getLayerAt(i-1));
            System.out.println();
        }
    }
}
