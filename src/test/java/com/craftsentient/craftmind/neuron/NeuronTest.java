package com.craftsentient.craftmind.neuron;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

public class NeuronTest {

    ArrayList<Double> inputs = new ArrayList<>();
    ArrayList<Double> weights = new ArrayList<>();
    Double BIAS = 1.0;

    @BeforeEach
    public void setUp() {
        inputs.add(1.0);
        weights.add(1.0);
    }

    @Test
    public void generateOutputTest() {
        Neuron neuron = Neuron.builder()
                .weights(weights)
                .bias(BIAS)
                .output(0.0)
                .build();
        Double output = neuron.generateOutput(inputs);

        Assertions.assertEquals(2.0, neuron.getOutput());
    }

    @Test
    public void createNeuronTest() {
        Neuron neuron = new Neuron(4, 2.0, 1, -1);
        Assertions.assertNotNull(neuron);
        Assertions.assertEquals(1, neuron.getMax());
        Assertions.assertEquals(-1, neuron.getMin());
        Assertions.assertEquals(2.0, neuron.getBias());
    }

    @Test
    public void generateNeuronTest() {
        Neuron neuron = new Neuron(4, 1.0);
        System.out.println(neuron.getWeights());
        Assertions.assertNotNull(neuron);
    }
}
