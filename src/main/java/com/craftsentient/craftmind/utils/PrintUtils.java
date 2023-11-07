package com.craftsentient.craftmind.utils;

import com.craftsentient.craftmind.layer.DenseLayer;
import com.craftsentient.craftmind.layers.DenseLayers;
import com.craftsentient.craftmind.neuron.Neuron;

import java.util.ArrayList;
import java.util.Map;
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
        System.out.println(backgroundBlack(green(bold(":::: " + data + " ::::"))));
    }
    public static void printSubTitle(String data) {
        System.out.println(backgroundBlack(cyan(bold(":::: " + data + " ::::"))));
    }
    public static void printGeneric(String data){
        System.out.println(cyan(bold("[LOG] ") + data));
    }
    public static void printGeneric(double data){
        System.out.println(cyan(bold("[LOG] ") + data));
    }
    public static void printGeneric(int data){
        System.out.println(cyan(bold("[LOG] ") + data));
    }
    public static void printGeneric(float data){
        System.out.println(cyan(bold("[LOG] ") + data));
    }
    public static void printGeneric(short data){
        System.out.println(cyan(bold("[LOG] ") + data));
    }
    public static void printGeneric(long data){
        System.out.println(cyan(bold("[LOG] ") + data));
    }
    public static void printGeneric(double[] vec, String label){
        System.out.println(bold(cyan(":::: " + label + " ::::")));
        AtomicInteger counter = new AtomicInteger(1);
        IntStream.range(0, vec.length).forEach(i-> {
            if(i != vec.length-1) System.out.print(cyan(bold(String.valueOf(i))) + ": " + vec[i] + ", ");
            else System.out.print(cyan(bold(String.valueOf(i))) + ": " + vec[i]);
        });
        System.out.print("\n");
    }
    public static void printGeneric(int[] vec, String label){
        System.out.println(bold(green(":::: " + label + " ::::")));
        AtomicInteger counter = new AtomicInteger(1);
        IntStream.range(0, vec.length).forEach(i-> {
            if(i != vec.length-1) System.out.print(cyan(bold(String.valueOf(i))) + ": " + vec[i] + ", ");
            else System.out.print(cyan(bold(String.valueOf(i))) + ": " + vec[i]);
        });
        System.out.print("\n");
    }
    public static void printGeneric(String[] vec, String label){
        System.out.println(bold(green(":::: " + label + " ::::")));
        AtomicInteger counter = new AtomicInteger(1);
        IntStream.range(0, vec.length).forEach(i-> {
            if(i != vec.length-1) System.out.print(cyan(bold(String.valueOf(i))) + ": " + vec[i] + ", ");
            else System.out.print(cyan(bold(String.valueOf(i))) + ": " + vec[i]);
        });
        System.out.print("\n");
    }
    public static void printGeneric(ArrayList<Object> vec, String label){
        System.out.println(bold(green(":::: " + label + " ::::")));
        AtomicInteger counter = new AtomicInteger(1);
        IntStream.range(0, vec.size()).forEach(i-> {
            if(i != vec.size()-1) System.out.print(cyan(bold(String.valueOf(i))) + ": " + vec.get(i) + ", ");
            else System.out.print(cyan(bold(String.valueOf(i))) + ": " + vec.get(i));
        });
        System.out.print("\n");
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
        System.out.println(bold(cyan(":::: " + label + " ::::")));
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


    // warning
    public static void printWarning(double data){
        System.out.println(yellow(bold("[WARNING] ") + data));
    }
    public static void printWarning(float data){
        System.out.println(yellow(bold("[WARNING] ") + data));
    }
    public static void printWarning(int data){
        System.out.println(yellow(bold("[WARNING] ") + data));
    }
    public static void printWarning(short data){
        System.out.println(yellow(bold("[WARNING] ") + data));
    }
    public static void printWarning(long data){
        System.out.println(yellow(bold("[WARNING] ") + data));
    }
    public static void printWarning(String data){
        System.out.println(yellow(bold("[WARNING] ") + data));
    }
    public static void printWarning(double[] vec){
        System.out.print(bold(yellow("[WARNING] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(yellow("]")));
            else if(i == 0) System.out.print(bold(yellow("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printWarning(int[] vec){
        System.out.print(bold(yellow("[WARNING] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(yellow("]")));
            else if(i == 0) System.out.print(bold(yellow("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printWarning(String[] vec){
        System.out.print(bold(yellow("[WARNING] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(yellow("]")));
            else if(i == 0) System.out.print(bold(yellow("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printWarning(double[] vec, String label){
        System.out.println(bold(yellow(":::: " + label + " ::::")));
        System.out.print(bold(yellow("[WARNING] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(yellow("]")));
            else if(i == 0) System.out.print(bold(yellow("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printWarning(int[] vec, String label){
        System.out.println(bold(yellow(":::: " + label + " ::::")));
        System.out.print(bold(yellow("[WARNING] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(yellow("]")));
            else if(i == 0) System.out.print(bold(yellow("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printWarning(String[] vec, String label){
        System.out.println(bold(yellow(":::: " + label + " ::::")));
        System.out.print(bold(yellow("[WARNING] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(yellow("]")));
            else if(i == 0) System.out.print(bold(yellow("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printWarning(ArrayList<Object> vec, String label){
        System.out.println(bold(yellow(":::: " + label + " ::::")));
        System.out.print(bold(yellow("[WARNING] ")));
        IntStream.range(0, vec.size()).forEach(i-> {
            if(i == vec.size()-1) System.out.print(vec.get(i) + bold(yellow("]")));
            else if(i == 0) System.out.print(bold(yellow("[")) + vec.get(i) + ", ");
            else System.out.print(String.valueOf(vec.get(i)) + ", ");
        });
        System.out.print("\n");
    }


    // negative
    public static void printNegative(float data){
        System.out.println(yellow(bold("[ERROR] ")) + data);
    }
    public static void printNegative(double data){
        System.out.println(yellow(bold("[ERROR] ")) + data);
    }
    public static void printNegative(int data){
        System.out.println(yellow(bold("[ERROR] ")) + data);
    }
    public static void printNegative(long data){
        System.out.println(yellow(bold("[ERROR] ")) + data);
    }
    public static void printNegative(short data){
        System.out.println(yellow(bold("[ERROR] ")) + data);
    }
    public static void printNegative(String data){
        System.out.println(yellow(bold("[ERROR] ")) + data);
    }
    public static void printNegative(String info, double data){
        System.out.println(yellow(bold("[ERROR] ")) + info + data);
    }
    public static void printNegative(String info, int data){
        System.out.println(yellow(bold("[ERROR] ")) + info + data);
    }
    public static void printNegative(String info, long data){
        System.out.println(yellow(bold("[ERROR] ")) + info + data);
    }
    public static void printNegative(double[] vec){
        System.out.print(bold(red("[ERROR] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printNegative(int[] vec){
        System.out.print(bold(red("[ERROR] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printNegative(String[] vec){
        System.out.print(bold(red("[ERROR] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printNegative(double[] vec, String label){
        System.out.println(bold(red(":::: " + label + " ::::")));
        System.out.print(bold(red("[ERROR] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printNegative(int[] vec, String label){
        System.out.println(bold(red(":::: " + label + " ::::")));
        System.out.print(bold(red("[ERROR] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printNegative(String[] vec, String label){
        System.out.println(bold(red(":::: " + label + " ::::")));
        System.out.print(bold(red("[ERROR] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printNegative(ArrayList<Object> vec, String label){
        System.out.println(bold(red(":::: " + label + " ::::")));
        System.out.print(bold(red("[ERROR] ")));
        IntStream.range(0, vec.size()).forEach(i-> {
            if(i == vec.size()-1) System.out.print(vec.get(i) + bold(red("]")));
            else if(i == 0) System.out.print(bold(red("[")) + vec.get(i) + ", ");
            else System.out.print(String.valueOf(vec.get(i)) + ", ");
        });
        System.out.print("\n");
    }


    // info
    public static void printInfo(String data){
        System.out.println(blue(bold("[INFO] ")) + data);
    }
    public static void printInfo(double data){ System.out.println(blue(bold("[INFO] ")) + data); }
    public static void printInfo(int data){
        System.out.println(blue(bold("[INFO] ")) + data);
    }
    public static void printInfo(long data){
        System.out.println(blue(bold("[INFO] ")) + data);
    }
    public static void printInfo(float data){
        System.out.println(blue(bold("[INFO] ")) + data);
    }
    public static void printInfo(short data){
        System.out.println(blue(bold("[INFO] ")) + data);
    }
    public static void printInfo(String info, double data){
        System.out.println(blue(bold("[INFO] ")) + info + " " + data);
    }
    public static void printInfo(String info, int data){
        System.out.println(blue(bold("[INFO] ")) + info + " " + data);
    }
    public static void printInfo(String info, long data){
        System.out.println(blue(bold("[INFO] ")) + info + " " + data);
    }
    public static void printInfo(String info, String data){
        System.out.println(info + " " + data);
    }
    public static void printInfo(double[] vec){
        System.out.print(bold(blue("[INFO] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printInfo(int[] vec){
        System.out.print(bold(blue("[INFO] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printInfo(String[] vec){
        System.out.print(bold(blue("[INFO] ")));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printInfo(String info, double[] vec){
        System.out.print(bold(blue("[INFO] ") + info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printInfo(String info, int[] vec){
        System.out.print(bold(blue("[INFO] ") + info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printInfo(String info, String[] vec){
        System.out.print(bold(blue("[INFO] ") + info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void printInfo(String info, ArrayList<Object> vec){
        System.out.print(bold(blue("[INFO] ") + info + " "));
        IntStream.range(0, vec.size()).forEach(i-> {
            if(i == vec.size()-1) System.out.print(vec.get(i) + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec.get(i) + ", ");
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
    public static void print(double[] vec){
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(int[] vec){
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String[] vec){
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, double[] vec){
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, int[] vec){
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, String[] vec){
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.length).forEach(i-> {
            if(i == vec.length-1) System.out.print(vec[i] + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec[i] + ", ");
            else System.out.print(String.valueOf(vec[i]) + ", ");
        });
        System.out.print("\n");
    }
    public static void print(String info, ArrayList<Object> vec){
        System.out.print(bold(info + " "));
        IntStream.range(0, vec.size()).forEach(i-> {
            if(i == vec.size()-1) System.out.print(vec.get(i) + bold(blue("]")));
            else if(i == 0) System.out.print(blue(bold("[")) + vec.get(i) + ", ");
            else System.out.print(String.valueOf(vec.get(i)) + ", ");
        });
        System.out.print("\n");
    }


    // generic matrix
    public static void printGeneric(double[][] matrix, String label){
        System.out.println(bold(green(":::: " + label + " ::::")));
        IntStream.range(0, matrix.length).forEach(i-> {
            IntStream.range(0, matrix[0].length).forEach(j -> {
                if(j == 0) System.out.print(blue(bold("[")) + matrix[i][j] + ", ");
                else if(j == matrix[0].length-1) System.out.print(matrix[i][j] + blue(bold("]")));
                else System.out.print(matrix[i][j] + ", ");
            });
            if(i != matrix.length - 1) System.out.print(",");
        });
        System.out.print("\n");
    }
    public static void printGeneric(int[][] matrix, String label){
        System.out.println(bold(green(":::: " + label + " ::::")));
        IntStream.range(0, matrix.length).forEach(i-> {
            IntStream.range(0, matrix[0].length).forEach(j -> {
                if(j == 0) System.out.print(blue(bold("[")) + matrix[i][j] + ", ");
                else if(j == matrix[0].length-1) System.out.print(matrix[i][j] + blue(bold("]")));
                else System.out.print(matrix[i][j] + ", ");
            });
            if(i != matrix.length - 1) System.out.print(",");
        });
        System.out.print("\n");
    }
    public static void printGeneric(String[][] matrix, String label){
        System.out.println(bold(green(":::: " + label + " ::::")));
        IntStream.range(0, matrix.length).forEach(i-> {
            IntStream.range(0, matrix[0].length).forEach(j -> {
                if(j == 0) System.out.print(blue(bold("[")) + matrix[i][j] + ", ");
                else if(j == matrix[0].length-1) System.out.print(matrix[i][j] + blue(bold("]")));
                else System.out.print(matrix[i][j] + ", ");
            });
            if(i != matrix.length - 1) System.out.print(",");
        });
        System.out.print("\n");
    }

    // info matrix
    public static void printInfo(double[][] matrix){
        System.out.print(bold(blue("[INFO] ")));
        IntStream.range(0, matrix.length).forEach(i-> {
            IntStream.range(0, matrix[0].length).forEach(j -> {
                if(j == 0) System.out.print(blue(bold("[")) + matrix[i][j] + ", ");
                else if(j == matrix[0].length-1) System.out.print(matrix[i][j] + blue(bold("]")));
                else System.out.print(matrix[i][j] + ", ");
            });
            if(i != matrix.length - 1) System.out.print(",");
        });
        System.out.print("\n");
    }
    public static void printInfo(int[][] matrix){
        System.out.print(bold(blue("[INFO] ")));
        IntStream.range(0, matrix.length).forEach(i-> {
            IntStream.range(0, matrix[0].length).forEach(j -> {
                if(j == 0) System.out.print(blue(bold("[")) + matrix[i][j] + ", ");
                else if(j == matrix[0].length-1) System.out.print(matrix[i][j] + blue(bold("]")));
                else System.out.print(matrix[i][j] + ", ");
            });
            if(i != matrix.length - 1) System.out.print(",");
        });
        System.out.print("\n");
    }
    public static void printInfo(String[][] matrix){
        System.out.print(bold(blue("[INFO] ")));
        IntStream.range(0, matrix.length).forEach(i-> {
            IntStream.range(0, matrix[0].length).forEach(j -> {
                if(j == 0) System.out.print(blue(bold("[")) + matrix[i][j] + ", ");
                else if(j == matrix[0].length-1) System.out.print(matrix[i][j] + blue(bold("]")));
                else System.out.print(matrix[i][j] + ", ");
            });
            if(i != matrix.length - 1) System.out.print(",");
        });
        System.out.print("\n");
    }
    public static void printInfo(String info, double[][] matrix){
        System.out.print(bold(blue("[INFO] ") + info + " "));
        IntStream.range(0, matrix.length).forEach(i-> {
            IntStream.range(0, matrix[0].length).forEach(j -> {
                if(j == 0) System.out.print(blue(bold("[")) + matrix[i][j] + ", ");
                else if(j == matrix[0].length-1) System.out.print(matrix[i][j] + blue(bold("]")));
                else System.out.print(matrix[i][j] + ", ");
            });
            if(i != matrix.length - 1) System.out.print(",");
        });
        System.out.print("\n");
    }
    public static void printInfo(String info, int[][] matrix){
        System.out.print(bold(blue("[INFO] ") + info + " "));
        IntStream.range(0, matrix.length).forEach(i-> {
            IntStream.range(0, matrix[0].length).forEach(j -> {
                if(j == 0) System.out.print(blue(bold("[")) + matrix[i][j] + ", ");
                else if(j == matrix[0].length-1) System.out.print(matrix[i][j] + blue(bold("]")));
                else System.out.print(matrix[i][j] + ", ");
            });
            if(i != matrix.length - 1) System.out.print(",");
        });
        System.out.print("\n");
    }
    public static void printInfo(String info, String[][] matrix) {
        System.out.print(bold(blue("[INFO] ") + info + " "));
        IntStream.range(0, matrix.length).forEach(i -> {
            IntStream.range(0, matrix[0].length).forEach(j -> {
                if (j == 0) System.out.print(blue(bold("[")) + matrix[i][j] + ", ");
                else if (j == matrix[0].length - 1) System.out.print(matrix[i][j] + blue(bold("]")));
                else System.out.print(matrix[i][j] + ", ");
            });
            if (i != matrix.length - 1) System.out.print(",");
        });
        System.out.print("\n");
    }
    public static <K, V> void printInfo(Map<K, V> map){
        System.out.print(bold(blue("[INFO] ")));
        IntStream.range(0, map.size()).forEach(i-> {
            System.out.print(String.valueOf(i) + map.get(i));
        });
        System.out.print("\n");
    }
    public static <Integer, V> void printInfo(String info, Map<Integer, V> map){
        System.out.print(bold(blue("[INFO] ")) + info + " ");
        map.forEach((i, v) -> {
            if (map.size() == 1) {
                System.out.print(blue(bold("<")) + i + ": " + v + blue(bold(">")));
                System.out.print("\n");
                return;
            }
            if (i.equals(0)) System.out.print(blue(bold("<")) + i + ": " + v + ",  ");
            else if (i.equals(map.size()-1)) System.out.print(i + ": " + v + blue(bold(">")));
            else System.out.print(i + ": " + map.get(i) + ",  ");
        });
        System.out.print("\n");
    }

    public static void printNeurons(String label, Neuron neuron){
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

    public static void printLayers(String label, DenseLayers layers) {
        System.out.println(bold(green(":::: " + label + " NETWORK ::::")));
        for(int i = 1; i <= layers.getLayerList().size(); i++){
            printLayer("" + i, layers.getLayerAt(i-1));
            System.out.println();
        }
    }



}
