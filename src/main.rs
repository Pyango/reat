use std::collections::HashMap;

struct Genome<'a> {
    compatibility_disjoint_coefficient: u32,
    adjusted_fitness: Option<f32>,
    neuron_add_prob: f32,
    neuron_delete_prob: f32,
    conn_add_prob: f32,
    conn_delete_prob: f32,
    generation: i32,
    ancestors: Option<[&'a Genome<'a>;2]>,
}

struct Specie<'a> {
    key: String,
    genomes: HashMap<i32, &'a Genome<'a>>,
}

struct Population<'a> {
    last_species_count: i32,
    size: i32,
    initial_fitness: i32,
    nr_offsprings: i32,
    fitness_threshold: i32,
    num_inputs: i32,
    num_outputs: i32,
    genomes: Option<HashMap<i32, &'a Genome<'a>>>,  // use i32 because genomes can have negative indexes
    species: Option<HashMap<u32, &'a Specie<'a>>>,  // Use u32 because its only positive numbers
}

impl Default for Population<'_> {
    fn default() -> Self {
        Population {
            last_species_count: 2,
            size: 0,
            initial_fitness: 0,
            nr_offsprings: 0,
            fitness_threshold: 0,
            num_inputs: 0,
            num_outputs: 0,
            genomes: None,
            species: None,
        }
    }
}

// impl Population {
//     fn new() -> Population {
//         Population {
//             last_species_count: 0,
//             nr_offsprings: 0.5,
//         }
//     }
// }

fn main() {
    println!("Hello, world!");
}
