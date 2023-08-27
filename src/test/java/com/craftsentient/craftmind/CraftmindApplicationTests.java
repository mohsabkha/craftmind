package com.craftsentient.craftmind;

import com.craftsentient.craftmind.layer.Layer;
import com.craftsentient.craftmind.neuron.Neuron;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.ArrayList;

@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest(){
        // create neuron
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

        // create neuron
        Neuron neuron3 = new Neuron();
        neuron3.getWeights().add(-0.26);
        neuron3.getWeights().add(-0.27);
        neuron3.getWeights().add(0.17);
        neuron3.getWeights().add(0.87);
        neuron3.setBias(0.5);

        // create layer
        Layer layer = new Layer();

        // add inputs
        layer.addInput(1.0);
        layer.addInput(2.0);
        layer.addInput(3.0);
        layer.addInput(2.5);

        // add neuron
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        // generate and store outputs
        ArrayList<Double> layerOutputs;
        layerOutputs = layer.generateLayerOutput();

        //generate
        Neuron neuron = new Neuron(4, 1.0);
        System.out.println(neuron.getWeights());
        Assertions.assertNotNull(neuron);

        // test against expected output
        Assertions.assertEquals(4.8, layerOutputs.get(0));
        Assertions.assertEquals(1.21, layerOutputs.get(1));
        Assertions.assertEquals(2.385, layerOutputs.get(2));
    }

    @Test
    public void mainTest(){
        String [] args = new String[0];
        CraftmindApplication.main(args);
    }
}
