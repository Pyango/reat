use std::cell::{RefCell};
use std::collections::HashMap;
use std::sync::Arc;
use ndarray::{ArrayView, Ix2};
use crate::{specie};
use crate::genome::Genome;
use rand::seq::SliceRandom;
use bincode::{Decode, Encode};
use crate::helpers::generate_uuid_key;

const SURVIVAL_THRESHOLD: i32 = 20;

#[derive(Default, Encode, Decode, PartialEq, Debug)]
pub struct Population {
    pub input_shape: RefCell<Vec<usize>>,
    pub output_shape: RefCell<Vec<usize>>,
    pub window_shape: RefCell<Vec<usize>>,
    pub stride: RefCell<usize>,
    pub size: i32,
    last_species_count: i32,
    survival_threshold: i32,
    compatibility_threshold: f32,
    max_species: i32,
    compatibility_threshold_mutate_power: f32,
    nr_offsprings: i32,
    fitness_threshold: f32,
    pub genomes: RefCell<HashMap<String, Arc<Genome>>>,
    pub best: RefCell<Arc<Genome>>,
    species: RefCell<Vec<specie::Specie>>,
}

impl Population {
    pub fn new(input_shape: Vec<usize>, window_shape: Vec<usize>, output_shape: Vec<usize>, stride: usize, size: i32, fitness_threshold: f32) -> Self {
        let p = Population {
            last_species_count: 2,
            genomes: RefCell::new(HashMap::new()),
            survival_threshold: SURVIVAL_THRESHOLD,
            input_shape: RefCell::new(input_shape),
            window_shape: RefCell::new(window_shape),
            output_shape: RefCell::new(output_shape),
            stride: RefCell::new(stride),
            fitness_threshold,
            ..Population::default()
        };
        let base_genome = p.create_genome();

        for _i in 1..size {
            let genome = base_genome.deep_clone();
            let mut new_key = generate_uuid_key();
            new_key = genome.set_key(new_key.clone());
            p.genomes.borrow_mut().insert(new_key.clone(), Arc::new(genome));
        }
        p
    }
    fn create_genome(&self) -> Arc<Genome> {
        let genome = Arc::new(Genome::new(self.input_shape.borrow().clone(), self.window_shape.borrow().clone(), self.stride.borrow().clone(), self.output_shape.borrow().clone()));
        self.genomes.borrow_mut().insert(genome.key.borrow().clone(), genome.clone());
        genome
    }
    fn mean_squared_error(&self, y: &[f32], y_hat: &[f32]) -> Result<f32, &'static str> {
        if y.len() != y_hat.len() {
            return Err("Vectors must have the same length");
        }

        let mse = y.iter().zip(y_hat.iter())
            .map(|(y_i, y_hat_i)| (y_i - y_hat_i).powi(2))
            .sum::<f32>() / y.len() as f32;

        Ok(mse)
    }
    fn simple_average(&self, fitness_values: &[f32]) -> f32 {
        fitness_values.iter().sum::<f32>() / fitness_values.len() as f32
    }
    fn exponential_moving_average(&self, fitness_values: &[f32], alpha: f32) -> f32 {
        let mut ema = fitness_values[0];
        for &value in &fitness_values[1..] {
            ema = alpha * value + (1.0 - alpha) * ema;
        }
        ema
    }
    pub fn activate(&self, generation: &i32, xvec_2d: &Vec<Vec<f32>>, yvec_2d: &Vec<Vec<f32>>) {
        let genomes = self.genomes.borrow().clone();
        for g in genomes.values() {
            g.set_generation(&generation);
            {
                *g.fitness.borrow_mut() = 0.0;
            }
            let mut fitness_values: Vec<f32> = vec![];
            for (index, row) in xvec_2d.iter().enumerate() {
                let row_vec: Vec<f32> = row.iter().cloned().collect();
                let outputs = g.activate(&row_vec);
                // println!("{:?}, {:?}", &outputs, &yvec_2d[index]);
                let mse = self.mean_squared_error(&yvec_2d[index], &outputs).unwrap();
                if mse.is_nan() {
                    g.show(index.to_string()).unwrap();
                }
                fitness_values.push(1.0 / (1.0 + mse));
            }
            // println!("fitness_values, {:?}, avg: {:?}", fitness_values, self.simple_average(&fitness_values));
            *g.fitness.borrow_mut() = self.simple_average(&fitness_values);
        }
    }

    pub fn mutate(&self, generation: i32) -> bool {
        let genomes = self.genomes.borrow().clone();
        let mut best: Option<&Genome> = None;
        for genome in genomes.values() {
            match best {
                Some(b) => {
                    if genome.fitness > b.fitness {
                        best = Some(genome);
                        *self.best.borrow_mut() = Arc::clone(genome);
                    }
                },
                None => best = Some(genome),
            }
        }
        println!("Generation {}", generation);
        println!("And the best genome is: {} with a fitness of {} and generation {}", best.unwrap().key.borrow(), *best.unwrap().fitness.borrow(), best.unwrap().generation.borrow());
        println!("Number of neurons: {} and {} connections.", best.unwrap().neurons.len() + best.unwrap().input.len() + best.unwrap().output.len(), best.unwrap().connections.len());
        println!("neuron_add_prob {}", best.unwrap().neuron_add_prob.borrow());
        println!("neuron_delete_prob {}", best.unwrap().neuron_delete_prob.borrow());
        println!("conn_add_prob {}", best.unwrap().conn_add_prob.borrow());
        println!("conn_delete_prob {}", best.unwrap().conn_delete_prob.borrow());
        println!("Genomes {}", genomes.len());
        // let pretty_string = serde_json::to_string_pretty(&best.unwrap()).unwrap();
        // println!("{}", pretty_string);
        if *best.unwrap().fitness.borrow() >= self.fitness_threshold {
            best.unwrap().show("".to_string()).unwrap();
            return true;
        }
        let mut top_genomes: Vec<&Arc<Genome>> = genomes.values().collect();
        top_genomes.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        let mut bad_genomes: Vec<&Arc<Genome>> = genomes.values().collect();
        bad_genomes.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal));
        // Mutate
        let start = (top_genomes.len() as f32 * 0.1) as usize;
        let end = (top_genomes.len() as f32 * 0.2) as usize;
        for genome in &mut top_genomes[start..end] {
            // println!("{}", genome.fitness.borrow());
            genome.mutate(&generation);
        }
        // Crossover
        // let crossover_genomes: Vec<&Arc<Genome>> = top_genomes.iter().take((top_genomes.len() as f32 * 0.1) as usize).map(|&genome| genome).collect();
        // let genomes_to_delete: Vec<&Arc<Genome>> = bad_genomes.iter().skip((bad_genomes.len() as f32 * 0.1) as usize).take((bad_genomes.len() as f32 * 0.1) as usize).map(|&genome| genome).collect();
        // // Crossover some genomes and take them after the worst 10% of bad genomes to replace their place so we keep the number of genomes constant
        // for bad_genome in genomes_to_delete {
        //     if let (Some(parent1), Some(parent2)) = (crossover_genomes.choose(&mut rand::thread_rng()), crossover_genomes.choose(&mut rand::thread_rng())) {
        //         if *parent1.key.borrow() == *parent2.key.borrow() {
        //             continue;
        //         }
        //         let new_genome = self.create_genome();
        //         new_genome.crossover(parent1, parent2);
        //         // new_genome.mutate();
        //         self.genomes.borrow_mut().remove(&bad_genome.key.borrow().clone()); // Remove the bad genome from the IndexMap
        //     }
        // }
        // Delete the worst 10% of genomes and breed new ones
        let start = (bad_genomes.len() as f32 * 0.0) as usize;
        let end = (bad_genomes.len() as f32 * 0.1) as usize;
        // println!("Worst genome fitnesses");
        for genome in &mut bad_genomes[start..end] {
            // println!("{}", genome.fitness.borrow());
            if *genome.generation.borrow() >= self.survival_threshold {
                self.genomes.borrow_mut().remove(&genome.key.borrow().clone()); // Remove the bad genome from the IndexMap
                self.create_genome();
            } else {
                genome.mutate(&generation);
            }
        }

        return false;
    }

    pub fn train(&self, x: ArrayView<f32, Ix2>, y : ArrayView<f32, Ix2>, generations: i32) {
        let x_view: ArrayView<f32, Ix2> = x.view();
        let xvec_2d: Vec<Vec<f32>> = x_view
            .axis_iter(ndarray::Axis(0))
            .map(|row| row.to_vec())
            .collect();
        let y_view: ArrayView<f32, Ix2> = y.view();
        let yvec_2d: Vec<Vec<f32>> = y_view
            .axis_iter(ndarray::Axis(0))
            .map(|row| row.to_vec())
            .collect();

        for generation in 0..generations {
            self.activate(&generation, &xvec_2d, &yvec_2d);
            // When mutate returns true we break the training because we crossed the fitness threshold
            if self.mutate(generation) {
                break;
            }
        }
    }
}
