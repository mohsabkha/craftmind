package com.craftsentient.craftmind.neuron;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;

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
}
