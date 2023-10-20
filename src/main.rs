mod neuron;
mod genome;
mod specie;
mod population;
mod connection;
mod attribute;
mod activations;

use crate::population::Population;

extern crate serde;
extern crate serde_json;

extern crate zmq;
extern crate ndarray;

use ndarray::prelude::*;

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() {
    let dataset = array![
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ];

    // Slice features (X) and labels (y)
    let x = dataset.slice(s![.., 0..2]);
    let y = dataset.slice(s![.., 2..3]);

    let p = Population::new(2, 1, 100, 0.999);
    p.train(x, y, 10000);
    // println!("{:#?}", p);
}