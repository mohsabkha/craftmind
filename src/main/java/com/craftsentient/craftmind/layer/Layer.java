package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;

@Data
@Builder
public class Layer {
    ArrayList<Neuron> neuronList;
}
