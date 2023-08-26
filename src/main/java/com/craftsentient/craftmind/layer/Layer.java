package com.craftsentient.craftmind.layer;

import com.craftsentient.craftmind.neuron.Neuron;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Layer {
    ArrayList<Neuron> neuronList;
}
