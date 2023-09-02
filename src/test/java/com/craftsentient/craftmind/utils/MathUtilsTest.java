package com.craftsentient.craftmind.utils;

import com.craftsentient.craftmind.neuron.Neuron;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

public class MathUtilsTest {

    @Test
    public void mathUtilsConstructorTest(){
        MathUtils utils = new MathUtils();
        Assertions.assertNotNull(utils);
    }

    @Test
    public void addVectorsTest(){
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
    public void addVectorsDifferentSizesExceptionTest(){
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
    public void addVectorsSizeZeroException(){
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
}
