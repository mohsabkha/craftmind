package com.craftsentient.craftmind.neuron;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.ArrayList;

public class NeuronTest {

    ArrayList<Long> inputs = new ArrayList<>();
    ArrayList<Long> weights = new ArrayList<>();
    Long BIAS = 1L;

    @BeforeEach
    public void setUp(){
        inputs.add(1L);
        weights.add(1L);
    }

    @Test
    public void generateOutputTest(){
        Neuron neuron = Neuron.builder()
                .inputs(inputs)
                .weights(weights)
                .bias(BIAS)
                .output(0L)
                .build();
        Long output = neuron.generateOutput();

        Assertions.assertEquals(2, output);
    }
}
