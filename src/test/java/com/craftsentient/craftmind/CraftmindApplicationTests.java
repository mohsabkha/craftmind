package com.craftsentient.craftmind;

import com.craftsentient.craftmind.layer.Layer;
import com.craftsentient.craftmind.neuron.Neuron;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.Arrays;


@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() {
        // create neuron
        Neuron neuron1 = new Neuron();
        neuron1.addWeight(0.2);
        neuron1.addWeight(0.8);
        neuron1.addWeight(-0.5);
        neuron1.addWeight(1.0);
        neuron1.setBias(2.0);

        // create neuron
        Neuron neuron2 = new Neuron();
        neuron2.addWeight(0.5);
        neuron2.addWeight(-0.91);
        neuron2.addWeight(0.26);
        neuron2.addWeight(-0.5);
        neuron2.setBias(3.0);

        // create neuron
        Neuron neuron3 = new Neuron();
        neuron3.addWeight(-0.26);
        neuron3.addWeight(-0.27);
        neuron3.addWeight(0.17);
        neuron3.addWeight(0.87);
        neuron3.setBias(0.5);

        // create layer
        Layer layer = new Layer();
        Layer layer1 = new Layer();

        // add inputs
        layer.addInput(1.0);
        layer.addInput(2.0);
        layer.addInput(3.0);
        layer.addInput(2.5);


        // add neuron
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        Assertions.assertEquals(3, layer.getNeuronList().size());

        // generate and store outputs
        double[] layerOutputs;
        layerOutputs = layer.generateLayerOutput(); //

        layer1.addInput(layer);
        layer1.addNeuron(neuron1);
        layer1.addNeuron(neuron2);
        layer1.addNeuron(neuron3);
        double[]layer1Outputs;
        //layer1Outputs = layer1.generateLayerOutput();

        Arrays.stream(layer.getLayerOutputs()).forEach(i -> System.out.println(i));
        // test against expected output
        Assertions.assertEquals(3, layer.getLayerOutputs().length);
        Assertions.assertEquals(4.8, layerOutputs[0]);
        Assertions.assertEquals(1.21, layerOutputs[1]);
        Assertions.assertEquals(2.385, layerOutputs[2]);
    }
    @Test
    public void mainTest() {
        String [] args = new String[0];
        CraftmindApplication.main(args);
    }
}
