package com.craftsentient.craftmind;

import com.craftsentient.craftmind.layer.Layer;
import com.craftsentient.craftmind.neuron.Neuron;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.Arrays;


@SpringBootTest
class CraftmindApplicationTests {
    @Test
    public void neuralNetworkTest() {
        double[][] weights = {
                {    0.2, 0.8, -0.5, 1.0    },
                {    0.5,-0.91,0.26,-0.5    },
                {   -0.26,-0.27,0.17,0.87   }};
        // create neurons
        Neuron neuron1 = new Neuron(weights[0], 2.0);
        Neuron neuron2 = new Neuron(weights[1], 3.0);
        Neuron neuron3 = new Neuron(weights[2], 0.5);

        // create layer
        Layer layer1 = new Layer();
        Layer layer2 = new Layer();
        Layer layer3 = new Layer();
        Layer hiddenLayer = new Layer();

        layer1.addNeuron(neuron1);
        layer1.addNeuron(neuron2);
        layer1.addNeuron(neuron3);

        layer2.addNeuron(neuron1);
        layer2.addNeuron(neuron2);
        layer2.addNeuron(neuron3);

        layer3.addNeuron(neuron1);
        layer3.addNeuron(neuron2);
        layer3.addNeuron(neuron3);

        // add inputs
        layer1.addInput(1.0);
        layer1.addInput(2.0);
        layer1.addInput(3.0);
        layer1.addInput(2.5);

        layer2.addInput(2.0);
        layer2.addInput(5.0);
        layer2.addInput(-1.0);
        layer2.addInput(2.0);

        layer3.addInput(-1.5);
        layer3.addInput(2.7);
        layer3.addInput(3.3);
        layer3.addInput(-0.8);

        hiddenLayer.useOutputFromPreviousLayerAsInput(layer1);


        // created a matrix - each neuron has 4 weights and 4 inputs attached to those weights (3 x 4 matrices)


        hiddenLayer.addNeuron(new Neuron(3, 2.5));
        hiddenLayer.addNeuron(new Neuron(3, 1));
        hiddenLayer.addNeuron(new Neuron(3, 2));

        Assertions.assertEquals(3, layer1.getNeuronList().size());

        // generate and store outputs
        double[][] layerOutputs = new double[4][4];
        layerOutputs[0] = layer1.generateLayerOutput();
        layerOutputs[1] = layer2.generateLayerOutput();
        layerOutputs[2] = layer3.generateLayerOutput();
        layerOutputs[3] = hiddenLayer.generateLayerOutput();

        // test against expected output
        Assertions.assertEquals(3, layer1.getLayerOutputs().length);
        Assertions.assertEquals(4.8, layerOutputs[0][0]);
        Assertions.assertEquals(1.21, layerOutputs[0][1]);
        Assertions.assertEquals(2.385, layerOutputs[0][2]);

        for (int i = 0; i < layerOutputs.length; i++) {
            for (int j = 0; j < layerOutputs[0].length; j++) {
                System.out.print(layerOutputs[i][j] + " ");
            }
            System.out.println();
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
