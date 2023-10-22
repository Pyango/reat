extern crate chess;
extern crate ndarray;
extern crate serde;
extern crate serde_json;
extern crate zmq;




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
    let x = xor2.slice(s![.., 0..2]);
    let y = xor2.slice(s![.., 2..3]);

    let p = Population::new(2, 1, 1000, 0.999);
    p.train(x, y, 10000);

    let x_view: ArrayView<f32, Ix2> = x.view();
    let xvec_2d: Vec<Vec<f32>> = x_view
        .axis_iter(ndarray::Axis(0))
        .map(|row| row.to_vec())
        .collect();
    let y_view: ArrayView<f32, Ix2> = y.view();
    let _yvec_2d: Vec<Vec<f32>> = y_view
        .axis_iter(ndarray::Axis(0))
        .map(|row| row.to_vec())
        .collect();
    for (index, row) in xvec_2d.iter().enumerate() {
        let row_vec: Vec<f32> = row.iter().cloned().collect();
        let b = p.best.borrow();
        let _outputs = b.activate(&row_vec);
        b.show(format!("{}_", index));
    }

    // let xor3 = array![
    //     [0.0, 0.0, 0.0, 0.0],
    //     [0.0, 0.0, 1.0, 1.0],
    //     [0.0, 1.0, 0.0, 1.0],
    //     [0.0, 1.0, 1.0, 0.0],
    //     [1.0, 0.0, 0.0, 1.0],
    //     [1.0, 0.0, 1.0, 0.0],
    //     [1.0, 1.0, 0.0, 0.0],
    //     [1.0, 1.0, 1.0, 1.0],
    // ];
    //
    // // Slice features (X) and labels (y)
    // let x = xor3.slice(s![.., 0..3]);
    // let y = xor3.slice(s![.., 3..4]);
    //
    // let p = Population::new(3, 1, 1000, 0.999);
    // p.train(x, y, 10000);
    // let x_view: ArrayView<f32, Ix2> = x.view();
    // let xvec_2d: Vec<Vec<f32>> = x_view
    //     .axis_iter(ndarray::Axis(0))
    //     .map(|row| row.to_vec())
    //     .collect();
    // let y_view: ArrayView<f32, Ix2> = y.view();
    // let yvec_2d: Vec<Vec<f32>> = y_view
    //     .axis_iter(ndarray::Axis(0))
    //     .map(|row| row.to_vec())
    //     .collect();
    // for (index, row) in xvec_2d.iter().enumerate() {
    //     let row_vec: Vec<f32> = row.iter().cloned().collect();
    //     let b = p.best.borrow();
    //     let outputs = b.activate(&row_vec);
    //     b.show(format!("{}_", index));
    // }
}