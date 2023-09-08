package com.craftsentient.craftmind.matrix;

import com.craftsentient.craftmind.layer.Layer;
import com.craftsentient.craftmind.neuron.Neuron;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Matrix {
    @Getter
    @Setter
    private int rows;

    @Getter
    @Setter
    private int columns;

    @Getter
    @Setter
    private int[] dimensions = new int[2];

    @Getter
    @Setter
    private Double determinant;

    @Getter
    @Setter
    private ArrayList<Layer> data;

    public Matrix(){
        this.rows = 0;
        this.columns = 0;
        this.data = new ArrayList<>();
    }

    public Matrix(int rows, int cols){
        this.dimensions[0] = rows;
        this.dimensions[1] = cols;
        this.rows = rows;
        this.columns = cols;
        this.data = new ArrayList<>();

        // create row one by one then generate columns
        IntStream.range(0, cols).parallel().forEach( i -> {
            data.add(new Layer());
            data.get(i).generateLayer(cols);
        });
    }

    public Matrix(ArrayList<Layer> initial) {
        IntStream.range(0, initial.size()).parallel().forEach( i -> {
            if(i > 0 && initial.get(i).getNeuronList().size() != initial.get(i-1).getNeuronList().size()){
                throw new ArithmeticException("Matrix not uniform");
            }
        });
        this.data = initial;
        this.rows = initial.size();
        this.columns = initial.get(0).getNeuronList().size();
        this.dimensions[0] = this.rows;
        this.dimensions[1] = this.columns;
    }

    public void setNeuronAt(int row, int column, Neuron neuron) {
        data.get(row).getNeuronList().set(column, neuron);
    }

    public void setNeuronAt(int row, int column, int numberOfWeights, Double bias) {
        data.get(row).getNeuronList().set(column, new Neuron(numberOfWeights, bias));
    }

    public void setNeuronAt(int row, int column, int numberOfWeights, Double bias, int max, int min) {
        data.get(row).getNeuronList().set(column, new Neuron(numberOfWeights, bias, max, min));
    }

    public void setNeuronAt(int row, int column, ArrayList<Double> weights, Double bias) {
        data.get(row).getNeuronList().set(column, new Neuron(weights, bias));
    }

    public Neuron getNeuronAt(int row, int column) {
        return  data.get(row).getNeuronList().get(column);
    }

    public ArrayList<Neuron> getRow(int row) {
        return data.get(row).getNeuronList();
    }

    public Layer getRowLayer(int row) {
        return data.get(row);
    }

    public ArrayList<Neuron> getColumn(int column) {
        Layer layer = new Layer();
        data.stream().parallel().forEach(i -> layer.addNeuron(i.getNeuronList().get(column)));
        return layer.getNeuronList();
    }

    public Layer getColumnLayer(int column) {
        Layer layer = new Layer();
        data.stream().parallel().forEach(i -> layer.addNeuron(i.getNeuronList().get(column)));
        return layer;
    }

    public Neuron resetNeuronAt(int row, int column) {
        Neuron toRemove = this.getNeuronAt(row, column);
        this.setNeuronAt(row, column, new Neuron());
        return toRemove;
    }

    public void removeRow(int row) {
        data.remove(row);
        this.rows--;
    }

    public void removeColumn(int column) {
        data.stream().parallel().forEach(i -> i.getNeuronList().remove(column));
        this.columns--;
    }

    public void addColumn() {
        data.stream().parallel().forEach(i -> i.getNeuronList().add(new Neuron()));
        this.columns++;
    }

    public void addRow() {
        Layer layer = new Layer();
        layer.generateLayer(this.rows);
        data.add(layer);
        this.rows++;
    }

    public void scale(Double scaler) {
        this.data.stream().parallel().forEach(i -> i.getNeuronList().stream().parallel().forEach(x ->  x.setOutput(x.getOutput() * scaler)));
    }

    public void addMatrix(Matrix matrixToAdd) {
        if(this.rows != matrixToAdd.rows || this.columns != matrixToAdd.columns){
            throw new ArithmeticException("When adding matrices, rows and columns sizes need to be identical!");
        }
        AtomicInteger counter = new AtomicInteger();
        this.data = this.data.stream().parallel().map(i -> {
            return Layer.addLayer(i, matrixToAdd.getColumnLayer(counter.getAndIncrement()));
        }).collect(Collectors.toCollection(ArrayList::new));
    }

    public void subtractMatrix(Matrix matrixToAdd) {
        if(this.rows != matrixToAdd.rows || this.columns != matrixToAdd.columns){
            throw new ArithmeticException("When adding matrices, rows and columns sizes need to be identical!");
        }
        matrixToAdd.scale(-1.0);
        AtomicInteger counter = new AtomicInteger();
        this.data = this.data.stream().parallel().map(i -> {
            return Layer.addLayer(i, matrixToAdd.getColumnLayer(counter.getAndIncrement()));
        }).collect(Collectors.toCollection(ArrayList::new));;
    }


    public void inverse() {

    }


    public void transpose() {

    }


    public void exp() {

    }


    public void multiply() {

    }


    public void divide() {

    }


    public void generateDeterminant() {

    }


    // matrix util function
    public static boolean isDoubleMatrix(Object matrixData) {
        return
                matrixData instanceof ArrayList &&
                !((ArrayList<?>)matrixData).isEmpty()
                && ((ArrayList<?>)matrixData).get(0) instanceof Double;
    }
}
