package com.craftsentient.craftmind.utils;

import com.craftsentient.craftmind.neuron.Neuron;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

public class MathUtilsTest {

    @Test
    public void mathUtilsConstructorTest() {
        MathUtils utils = new MathUtils();
        Assertions.assertNotNull(utils);
    }

    @Test
    public void addVectorsTest() {
        Neuron neuron1 = new Neuron();
        neuron1.getWeights().add(0.2);
        neuron1.getWeights().add(0.8);
        neuron1.getWeights().add(-0.5);
        neuron1.getWeights().add(1.0);
        neuron1.setBias(2.0);

        // create neuron
        Neuron neuron2 = new Neuron();
        neuron2.getWeights().add(0.5);
        neuron2.getWeights().add(-0.91);
        neuron2.getWeights().add(0.26);
        neuron2.getWeights().add(-0.5);
        neuron2.setBias(3.0);

        ArrayList<Double> sum = MathUtils.addVectors(neuron1.getWeights(), neuron2.getWeights());
        Assertions.assertNotNull(sum);
        Assertions.assertFalse(sum.isEmpty());
    }

    @Test
    public void addVectorsDifferentSizesExceptionTest() {
        try {
            Neuron neuron1 = new Neuron();
            neuron1.getWeights().add(0.2);
            neuron1.getWeights().add(0.8);
            neuron1.getWeights().add(-0.5);
            neuron1.setBias(2.0);

            // create neuron
            Neuron neuron2 = new Neuron();
            neuron2.getWeights().add(0.5);
            neuron2.getWeights().add(-0.91);
            neuron2.getWeights().add(0.26);
            neuron2.getWeights().add(-0.5);
            neuron2.setBias(3.0);

            ArrayList<Double> sum = MathUtils.addVectors(neuron1.getWeights(), neuron2.getWeights());
        } catch( Exception e) {
            Assertions.assertEquals(e.getClass(), ArithmeticException.class);
            Assertions.assertEquals(e.getMessage(), "vectors should be of same size");
        }
    }

    @Test
    public void addVectorsSizeZeroException() {
        try {
            Neuron neuron1 = new Neuron();
            neuron1.setBias(2.0);

            // create neuron
            Neuron neuron2 = new Neuron();
            neuron2.setBias(3.0);

            ArrayList<Double> sum = MathUtils.addVectors(neuron1.getWeights(), neuron2.getWeights());
        } catch( Exception e) {
            Assertions.assertEquals(e.getClass(), ArithmeticException.class);
            Assertions.assertEquals(e.getMessage(), "vector sizes should not be zero");
        }
    }

    @Test
    public void dotProductTest(){
        Neuron neuron1 = new Neuron();
        neuron1.getWeights().add(1.0);
        neuron1.getWeights().add(2.0);
        neuron1.getWeights().add(3.0);
        neuron1.setBias(2.0);
        // create neuron
        Neuron neuron2 = new Neuron();
        neuron2.getWeights().add(3.0);
        neuron2.getWeights().add(4.0);
        neuron2.getWeights().add(5.0);
        neuron2.setBias(3.0);
        Double dotProduct1 = (Double)MathUtils.dotProduct(neuron1.getWeights(), neuron2.getWeights());
        Assertions.assertEquals(26.0, dotProduct1);

        Neuron neuron3 = new Neuron();
        neuron3.getWeights().add(1.0);
        neuron3.getWeights().add(2.0);
        neuron3.getWeights().add(3.0);
        neuron3.setBias(2.0);
        // create neuron
        Neuron neuron4 = new Neuron();
        neuron4.getWeights().add(4.0);
        neuron4.getWeights().add(5.0);
        neuron4.getWeights().add(6.0);
        neuron4.setBias(3.0);
        Double dotProduct2 = (Double)MathUtils.dotProduct(neuron4.getWeights(), neuron3.getWeights());
        Assertions.assertEquals(32.0, dotProduct2);



        ArrayList<ArrayList<Double>> matrix1 = new ArrayList<>();
        ArrayList<ArrayList<Double>> matrix2 = new ArrayList<>();

        ArrayList<Double> m1row1 = new ArrayList<>();
        m1row1.add(1.0);
        m1row1.add(2.0);
        m1row1.add(3.0);
        m1row1.add(4.0);

        ArrayList<Double> m1row2 = new ArrayList<>();
        m1row2.add(5.0);
        m1row2.add(6.0);
        m1row2.add(7.0);
        m1row2.add(8.0);

        ArrayList<Double> m2row1 = new ArrayList<>();
        m2row1.add(1.0);
        m2row1.add(2.0);
        m2row1.add(3.0);
        m2row1.add(4.0);

        ArrayList<Double> m2row2 = new ArrayList<>();
        m2row2.add(5.0);
        m2row2.add(6.0);
        m2row2.add(7.0);
        m2row2.add(8.0);

        matrix2.add(m2row1);
        matrix2.add(m2row2);

        matrix1.add(m1row1);
        matrix1.add(m1row2);
        System.out.println(matrix1);
        System.out.println(matrix2);


        ArrayList<Double> result = (ArrayList<Double>) MathUtils.dotProduct(matrix1, matrix2);

        System.out.println(result);
    }
}
