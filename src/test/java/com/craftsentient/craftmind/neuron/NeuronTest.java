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

    @Test
    public void NeuronConstructorTest(){

    }

    @Test
    public void createOutputCallTest(){
        NeuronImpl neuron = new NeuronImpl(inputs,weights,BIAS,0L);
        neuron.createOutput();

    }

    @Test
    public void createOutputWithInputsWeightsAndBiasTest(){
        Long output = NeuronImpl.createOutput(inputs, weights, BIAS);

    }

    @Test
    public void createOutputWithNeuronTest(){
        NeuronImpl neuron = new NeuronImpl(inputs,weights,BIAS,0L);
        neuron = NeuronImpl.createOutput(neuron);


    }
}
