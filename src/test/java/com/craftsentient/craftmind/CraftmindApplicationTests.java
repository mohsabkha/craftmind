package com.craftsentient.craftmind;

import com.craftsentient.craftmind.layer.Layer;
import com.craftsentient.craftmind.neuron.Neuron;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() {
        double[][] weights = {
                {    0.2, 0.8, -0.5, 1.0    },
                {    0.5,-0.91,0.26,-0.5    },
                {   -0.26,-0.27,0.17,0.87   }};

        double[][] inputs = {
                { 1,2,3,2.5 },
                { 2,5,-1,2 },
                { -1.5,2.7,3.3,-0.8 }
        };

        Layer layer1 = new Layer();
        layer1.addNeuron(new Neuron(weights[0], 2.0));
        layer1.addNeuron(new Neuron(weights[1], 3.0));
        layer1.addNeuron(new Neuron(weights[2], 0.5));
        layer1.addInput(inputs);

        Layer layer2 = new Layer();
        layer2.addNeuron(new Neuron(weights[0], 2.0));
        layer2.addNeuron(new Neuron(weights[1], 3.0));
        layer2.addNeuron(new Neuron(weights[2], 0.5));
        double[][] layerOutputs1 = (double[][])layer1.generateLayerOutput();

        layer2.useOutputFromPreviousLayerAsInput(layer1);
        double[][] layerOutputs2 = (double[][])layer2.generateLayerOutput();

        Assertions.assertEquals(3, layer1.getNeuronList().size());
        Assertions.assertEquals(3, layer1.getLayerOutputs().length);
        Assertions.assertEquals(4.8, layerOutputs1[0][0]);
        Assertions.assertEquals(1.21, layerOutputs1[0][1]);
        Assertions.assertEquals(2.385, layerOutputs1[0][2]);

        for (int i = 0; i < layerOutputs2.length; i++) {
            System.out.print("[");
            for (int j = 0; j < layerOutputs2[0].length; j++) {
                System.out.print(layerOutputs2[i][j] + ", ");
            }
            System.out.println("]");
        }
    }

    @Test
    public void mainTest() {
        String [] args = new String[0];
        CraftmindApplication.main(args);
    }

    @Test
    public void matrixTest(){

    }
}
