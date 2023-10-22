use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use ndarray::{ArrayView, Ix2};
use crate::{specie};
use crate::genome::Genome;
use rand::seq::SliceRandom;

const SURVIVAL_THRESHOLD: i32 = 10;

#[derive(Default, Debug)]
pub struct Population {
    num_inputs: i32,
    num_outputs: i32,
    size: i32,
    last_species_count: i32,
    survival_threshold: i32,
    compatibility_threshold: f32,
    max_species: i32,
    compatibility_threshold_mutate_power: f32,
    nr_offsprings: i32,
    fitness_threshold: f32,
    pub genomes: RefCell<HashMap<u32, Rc<Genome>>>,
    pub best: RefCell<Rc<Genome>>,
    species: RefCell<HashMap<u32, specie::Specie>>,
}

impl Population {
    pub fn new(num_inputs: i32, num_outputs: i32, size: i32, fitness_threshold: f32) -> Self {
        let p = Population {
            last_species_count: 2,
            genomes: RefCell::new(HashMap::new()),
            survival_threshold: SURVIVAL_THRESHOLD,
            num_inputs,
            num_outputs,
            fitness_threshold,
            ..Population::default()
        };
        for _i in 0..size {
            p.create_genome();
        }
        p
    }
    fn create_genome(&self) -> Rc<Genome> {
        let key = self.get_new_genome_key();
        let genome = Rc::new(Genome::new(key, self.num_inputs, self.num_outputs, self.size));
        self.genomes.borrow_mut().insert(key, Rc::clone(&genome));
        Rc::clone(&genome)
    }
    fn get_new_genome_key(&self) -> u32 {
        if !self.genomes.borrow().is_empty() {
            let max_key = self.genomes.borrow().keys().cloned().max().unwrap();
            return max_key + 1;
        }
        return 1;
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
    pub fn activate(&self, xvec_2d: &Vec<Vec<f32>>, yvec_2d: &Vec<Vec<f32>>) {
        let genomes = self.genomes.borrow().clone();
        for g in genomes.values() {
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
            genome.set_generation(&generation);
            match best {
                Some(b) => {
                    if genome.fitness > b.fitness {
                        best = Some(genome);
                        *self.best.borrow_mut() = Rc::clone(genome);
                    }
                },
                None => best = Some(genome),
            }
        }
        println!("Generation {}", generation);
        println!("And the best genome is: {} with a fitness of {} and generation {}", best.unwrap().key, *best.unwrap().fitness.borrow(), best.unwrap().generation.borrow());
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
        let mut top_genomes: Vec<&Rc<Genome>> = genomes.values().collect();
        top_genomes.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        let mut bad_genomes: Vec<&Rc<Genome>> = genomes.values().collect();
        bad_genomes.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal));
        let start = (top_genomes.len() as f32 * 0.0) as usize;
        let end = (top_genomes.len() as f32 * 0.3) as usize;
        // println!("top_genomes, {}", top_genomes.len());
        // println!("bad_genomes, {}", bad_genomes.len());

        // println!("Top genome fitnesses");
        for genome in &mut top_genomes[start..end] {
            // println!("{}", genome.fitness.borrow());
            genome.mutate();
        }
        let crossover_genomes: Vec<&Rc<Genome>> = top_genomes.iter().take((top_genomes.len() as f32 * 0.1) as usize).map(|&genome| genome).collect();
        let genomes_to_delete: Vec<&Rc<Genome>> = bad_genomes.iter().skip((bad_genomes.len() as f32 * 0.1) as usize).take((bad_genomes.len() as f32 * 0.1) as usize).map(|&genome| genome).collect();
        // Crossover some genomes and take them after the worst 10% of bad genomes to replace their place so we keep the number of genomes constant
        for bad_genome in genomes_to_delete {
            if let (Some(parent1), Some(parent2)) = (crossover_genomes.choose(&mut rand::thread_rng()), crossover_genomes.choose(&mut rand::thread_rng())) {
                let new_genome = self.create_genome(); // Assuming create_genome returns a mutable Genome
                new_genome.crossover(parent1, parent2); // Assuming you want to pass references to parent genomes.
                new_genome.set_generation(&0);
                new_genome.mutate();
                self.genomes.borrow_mut().remove(&bad_genome.key); // Remove the bad genome from the HashMap
            }
        }
        // Delete the worst 10% of genomes and breed new ones
        let start = (bad_genomes.len() as f32 * 0.0) as usize;
        let end = (bad_genomes.len() as f32 * 0.1) as usize;
        // println!("Worst genome fitnesses");
        for genome in &mut bad_genomes[start..end] {
            // println!("{}", genome.fitness.borrow());
            if *genome.generation.borrow() >= self.survival_threshold {
                self.genomes.borrow_mut().remove(&genome.key); // Remove the bad genome from the HashMap
                self.create_genome();
            } else {
                genome.set_generation(&0);
                genome.mutate()
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
            self.activate(&xvec_2d, &yvec_2d);
            // When mutate returns true we break the training because we crossed the fitness threshold
            if self.mutate(generation) {
                break;
            }
        }
    }
}
