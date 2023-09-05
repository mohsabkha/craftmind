package com.craftsentient.craftmind.neuralNetwork;


import com.craftsentient.craftmind.layer.Layer;
import org.apache.spark.SparkContext;
import org.apache.spark.rdd.RDD;

import java.util.logging.Level;

import java.util.ArrayList;
import java.util.logging.Logger;

public class NeuralNetwork {
    private ArrayList<Layer> layers;
    private ArrayList<ArrayList<Double>> batchInputs;
    private ArrayList<ArrayList<Double>> outputs;

    public void wordCount(){
        System.out.println("::INFO:: LOADING...");
        Logger.getLogger("org").setLevel(Level.INFO);
        System.out.println("::INFO:: LOADING... setting logger");
        SparkContext sc = new SparkContext("local[*]", "NeuralNetwork");
        System.out.println("::INFO:: LOADING... creating spark context");

        RDD<String> lines = sc.textFile("src/main/resources/ml-100k/u.data", 1);
        long numLines = lines.count();
        System.out.println("::INFO:: " + numLines);
    }
}
