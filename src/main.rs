extern crate chess;
extern crate ndarray;
extern crate serde;
extern crate serde_json;
extern crate zmq;

use std::any::Any;
use std::convert::TryInto;

use ndarray::prelude::*;

use crate::population::Population;

mod neuron;
mod genome;
mod specie;
mod population;
mod connection;
mod attribute;
mod activations;
mod chess_helpers;

fn main() {
    // let p = Population::new(64, 64, 100, 0.999);
    // for generation in 0..1000 {
    //     for g in p.genomes.borrow().values() {
    //         *g.fitness.borrow_mut() = 0.0;
    //         unsafe { chess_helpers::play_game(&g); }
    //         if generation == 0 {
    //             g.show("".to_string());
    //         }
    //      }
    //     p.mutate(generation);
    // }
    let xor2 = array![
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ];
    let xor3 = array![
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ];

    // Slice features (X) and labels (y)
    let x = xor3.slice(s![.., 0..3]);
    let y = xor3.slice(s![.., 3..4]);

    let p = Population::new(3, 1, 100, 0.999);
    p.train(x, y, 10000);
}