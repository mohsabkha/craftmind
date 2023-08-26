package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Builder;
import lombok.Data;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
@Data
@Builder
public class Layer {
    ArrayList<Neuron> neuronList;
}
