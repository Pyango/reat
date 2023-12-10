extern crate chess;
extern crate ndarray;
extern crate zmq;
extern crate uuid;
use std::fs::{metadata, OpenOptions};

use std::cell::RefCell;
use std::fs::File;
use bincode::encode_into_std_write;


use bincode::{config, Decode, Encode};
use ndarray::prelude::*;
use crate::genome::Genome;

use crate::population::Population;

mod neuron;
mod genome;
mod specie;
mod population;
mod connection;
mod attribute;
mod activations;
mod chess_helpers;
mod simple_float_rng;
mod ordered_ref_cell;
mod helpers;
mod t;

fn main() {
    // Example encode decode
    // let input_shape : Vec<usize> = vec![3, 3];
    // let window_shape : Vec<usize> = vec![2, 2];
    // let output_shape : Vec<usize> = vec![2];
    // let p = Population::new(input_shape, window_shape, output_shape, 1, 1, 1.0);
    // let config = config::standard();
    // let mut f = File::create("./foo.bar").unwrap();
    // encode_into_std_write(&p, &mut f, config).unwrap();
    // let mut f = File::open("./foo.bar").unwrap();
    // let decoded = bincode::decode_from_std_read(&mut f, config).unwrap();
    //
    // assert_eq!(p, decoded);
    // println!("{:?}", p);

    // Supervised learning example with chess engine
    // for generation in 0..10000 {
    //     let mut f = vec![];
    //     for g in p.genomes.borrow().values() {
    //         *g.fitness.borrow_mut() = 0.0;
    //         unsafe { chess_helpers::play_game(&g); }
    //         f.push(*g.fitness.borrow());
    //     }
    //     // println!("{:?}", f);
    //     p.mutate(generation);
    // }

    let config = config::standard();
    let xor2 = array![
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ];
    let x = xor2.slice(s![.., 0..2]);
    let y = xor2.slice(s![.., 2..3]);
    let file_existed = metadata("./xor2.model").is_ok();
    let mut xor2_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("./xor2.model")
        .unwrap();
    if !file_existed {
        let p = Population::new(vec![2], vec![1], vec![1], 1, 100, 0.999);
        let g = p.genomes.borrow().clone();
        let n = g.values().next().unwrap();
        p.train(x, y, 10);
        encode_into_std_write(&p, &mut xor2_file, config).unwrap();
    }
    let mut xor2_file = File::open("./xor2.model").unwrap();
    let decodede_p : Population = bincode::decode_from_std_read(&mut xor2_file, config).unwrap();
    decodede_p.train(x, y, 10000);
    let x_view: ArrayView<f32, Ix2> = x.view();
    let xvec_2d: Vec<Vec<f32>> = x_view
        .axis_iter(ndarray::Axis(0))
        .map(|row| row.to_vec())
        .collect();
    for (index, row) in xvec_2d.iter().enumerate() {
        let row_vec: Vec<f32> = row.iter().cloned().collect();
        let b = decodede_p.best.borrow();
        b.activate(&row_vec);
        b.show(format!("{}_", index));
    }
    let mut xor2_file = File::create("./xor2.model").unwrap();
    encode_into_std_write(&decodede_p, &mut xor2_file, config).unwrap();
    //
    //
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
    // for (index, row) in xvec_2d.iter().enumerate() {
    //     let row_vec: Vec<f32> = row.iter().cloned().collect();
    //     let b = p.best.borrow();
    //     b.activate(&row_vec);
    //     b.show(format!("{}_", index));
    // }
}