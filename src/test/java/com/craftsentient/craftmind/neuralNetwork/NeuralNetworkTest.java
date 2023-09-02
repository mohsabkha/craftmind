package com.craftsentient.craftmind.neuralNetwork;

import org.apache.hadoop.shaded.org.apache.commons.math3.linear.MatrixUtils;
import org.junit.Before;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class NeuralNetworkTest {
//    @Before
//    public void setUp(){
//        SparkConf sparkConf = new SparkConf().setAppName("3DMatrixMultiplication").setMaster("local");
//        JavaSparkContext sc = new JavaSparkContext(sparkConf);
//    }

    @Test
    public void testNeuralNetworkConstructor(){
        NeuralNetwork nn = new NeuralNetwork();
        Assertions.assertNotNull(nn);
    }
}
