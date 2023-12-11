use std::cell::RefCell;
use std::collections::{VecDeque};

use rand::seq::{IteratorRandom, SliceRandom};
use rand::{random, RngCore, thread_rng};
use crate::connection::{Connection};
use crate::neuron::Neuron;
use std::fs::File;
use std::io::prelude::*;
use std::rc::Rc;
use rand::distributions::uniform::SampleBorrow;
use bincode::{Decode, Encode};
use crate::helpers::generate_uuid_key;
use crate::ordered_ref_cell::OrderedRefCell;
use crate::t::Type;

const NEURON_ADD_PROB: f32 = 0.001;
const NEURON_DELETE_PROB: f32 = 0.001;
const CONN_ADD_PROB: f32 = 0.1;
const CONN_DELETE_PROB: f32 = 0.1;
// const NEURON_ADD_PROB: f32 = 0.05;
// const NEURON_DELETE_PROB: f32 = 0.01;
// const CONN_ADD_PROB: f32 = 0.05;
// const CONN_DELETE_PROB: f32 = 0.01;

#[derive(Encode, Decode, PartialEq, Debug, Clone)]
pub struct Genome {
    pub key: Rc<RefCell<String>>,
    pub input_shape: RefCell<Vec<usize>>,
    pub output_shape: RefCell<Vec<usize>>,
    pub window_shape: RefCell<Vec<usize>>,
    pub stride: usize,
    pub neuron_add_prob: RefCell<f32>,
    pub neuron_delete_prob: RefCell<f32>,
    pub conn_add_prob: RefCell<f32>,
    pub conn_delete_prob: RefCell<f32>,
    adjusted_fitness: f32,
    compatibility_disjoint_coefficient: f32,
    pub generation: RefCell<i32>,
    ancestors: RefCell<[String; 2]>,
    pub neurons: Rc<RefCell<OrderedRefCell<String, Neuron>>>,
    pub input: Rc<RefCell<OrderedRefCell<String, Neuron>>>,
    pub output: Rc<RefCell<OrderedRefCell<String, Neuron>>>,
    pub connections: Rc<RefCell<OrderedRefCell<(String, String), Connection>>>,
    pub fitness: RefCell<f32>,
}

impl Default for Genome {
    fn default() -> Self {
        Genome {
            key: Rc::new(RefCell::new(generate_uuid_key())),
            input: Rc::new(RefCell::new(OrderedRefCell::new())),
            neurons: Rc::new(RefCell::new(OrderedRefCell::new())),
            output: Rc::new(RefCell::new(OrderedRefCell::new())),
            input_shape: RefCell::new(vec![0]),
            output_shape: RefCell::new(vec![0]),
            window_shape: RefCell::new(vec![0]),
            stride: 1,
            neuron_add_prob: RefCell::new(NEURON_ADD_PROB),
            neuron_delete_prob: RefCell::new(NEURON_DELETE_PROB),
            conn_add_prob: RefCell::new(CONN_ADD_PROB),
            conn_delete_prob: RefCell::new(CONN_DELETE_PROB),
            adjusted_fitness: 0.0,
            compatibility_disjoint_coefficient: 0.0,
            generation: RefCell::new(0),
            ancestors: RefCell::new(["".to_string(), "".to_string()]),
            connections: Rc::new(RefCell::new(OrderedRefCell::new())),
            fitness: RefCell::new(0.0),
        }
    }
}

impl Genome {
    pub fn new(input_shape: Vec<usize>, window_shape: Vec<usize>, stride: usize, output_shape: Vec<usize>) -> Self {
        let _rng = thread_rng();
        let g = Genome {
            input_shape: RefCell::new(input_shape),
            window_shape: RefCell::new(window_shape),
            output_shape: RefCell::new(output_shape),
            stride,
            // neuron_add_prob: RefCell::new(rng.gen()),
            // neuron_delete_prob: RefCell::new(rng.gen()),
            // conn_add_prob: RefCell::new(rng.gen()),
            // conn_delete_prob: RefCell::new(rng.gen()),
            ..Genome::default()
        };
        g
    }

    pub fn initialize(&self) {
        let input_shape = self.input_shape.borrow().clone();
        let window_shape = self.window_shape.borrow().clone();
        let output_shape = self.output_shape.borrow().clone();

        let output_size = output_shape.clone().iter().fold(1, |acc, &x| acc * x);
        for _ in 0..output_size {
            self.create_output_neuron();
        }
        let hidden_shape: Vec<usize> = input_shape.iter().zip(window_shape.iter())
            .map(|(&dim, &k_dim)| (dim - k_dim) / self.stride + 1)
            .collect();

        // Iterate over the output tensor's shape
        let mut hidden_coordinates = vec![0; hidden_shape.len()];
        let mut input_coordinates = vec![0; input_shape.len()];
        'outer: loop {
            // Perform the convolution at the current index
            let mut kernel_coordinates = vec![0; window_shape.len()];
            // let hidden_index = self.calculate_index(&hidden_coordinates, &hidden_shape);
            let (hidden_key, _) = self.create_neuron();
            'inner: loop {
                // Calculate the input index
                for (i, (&k_i, &_s)) in kernel_coordinates.iter().zip(input_shape.iter()).enumerate() {
                    input_coordinates[i] = hidden_coordinates[i] * self.stride + k_i;
                }

                // let input_index = self.calculate_index(&input_coordinates, &self.input_shape.borrow());
                let (input_key, _) = self.create_input_neuron();
                self.create_connection(input_key, Type::Input, hidden_key.clone(), Type::Hidden, 0.5);

                // Move to the next index in the kernel
                for i in (0..kernel_coordinates.len()).rev() {
                    kernel_coordinates[i] += 1;
                    if kernel_coordinates[i] < window_shape[i] {
                        continue 'inner;
                    }
                    kernel_coordinates[i] = 0;
                }
                break;
            }
            for output_index in 0..output_size {
                let output_neuron = self.output.borrow().get_by_index(&output_index).unwrap();
                self.create_connection(hidden_key.clone(), Type::Hidden, output_neuron.borrow().key.clone(), Type::Output, 0.5);
            }

            // Move to the next index in the output tensor
            for i in (0..hidden_coordinates.len()).rev() {
                hidden_coordinates[i] += 1;
                if hidden_coordinates[i] < hidden_shape[i] {
                    continue 'outer;
                }
                hidden_coordinates[i] = 0;
            }

            // Check if we've completed iterating over the output tensor
            if hidden_coordinates.iter().zip(hidden_shape.iter()).all(|(&i, &dim)| i <= dim) {
                break;
            }
        }

    }
    pub fn clone(&self) -> Self {
        Genome {
            key: Rc::new(RefCell::new((*self.key.borrow()).clone())),
            input_shape: RefCell::new((*self.input_shape.borrow()).clone()),
            output_shape: RefCell::new((*self.output_shape.borrow()).clone()),
            window_shape: RefCell::new((*self.window_shape.borrow()).clone()),
            stride: self.stride,
            neuron_add_prob: RefCell::new(*self.neuron_add_prob.borrow()),
            neuron_delete_prob: RefCell::new(*self.neuron_delete_prob.borrow()),
            conn_add_prob: RefCell::new(*self.conn_add_prob.borrow()),
            conn_delete_prob: RefCell::new(*self.conn_delete_prob.borrow()),
            adjusted_fitness: self.adjusted_fitness,
            compatibility_disjoint_coefficient: self.compatibility_disjoint_coefficient,
            generation: RefCell::new(*self.generation.borrow()),
            ancestors: RefCell::new((*self.ancestors.borrow()).clone()),
            neurons: Rc::new(RefCell::new((*self.neurons.borrow()).clone())),
            input: Rc::new(RefCell::new((*self.input.borrow()).clone())),
            output: Rc::new(RefCell::new((*self.output.borrow()).clone())),
            connections: Rc::new(RefCell::new((*self.connections.borrow()).clone())),
            fitness: RefCell::new(*self.fitness.borrow()),
        }
    }
    pub fn set_key(&self, key: String) -> String {
        *self.key.borrow_mut() = key.clone();
        key
    }
    fn flat_index(&self, shape: Vec<usize>, index: &[usize]) -> usize {
        index.iter().zip(shape.iter().rev())
            .fold(0, |acc, (&i, &dim)| acc * dim + i)
    }
    fn adjacent_neighbors(&self, coordinates: &Vec<usize>, input_shape: &Vec<usize>) -> Vec<Vec<usize>> {
        let mut neighbors = Vec::new();

        for i in 0..coordinates.len() {
            for &offset in [-1, 1].iter() {
                if (coordinates[i] as i32 + offset) >= 0
                    && (coordinates[i] as i32 + offset) < input_shape[i] as i32
                {
                    let mut neighbor_coordinates = coordinates.clone();
                    neighbor_coordinates[i] = (coordinates[i] as i32 + offset) as usize;
                    neighbors.push(neighbor_coordinates);
                }
            }
        }
        neighbors
    }
    fn calculate_index(&self, coordinates: &Vec<usize>, input_shape: &Vec<usize>) -> usize {
        coordinates
            .iter()
            .enumerate()
            .fold(0, |acc, (i, &x)| acc * input_shape[i] + x)
    }
    fn increment_coordinates(&self, coordinates: &mut Vec<usize>, input_shape: &Vec<usize>) {
        let num_dimensions = coordinates.len();
        let mut i = num_dimensions - 1;

        while i >= 0 {
            coordinates[i] += 1;

            if coordinates[i] >= input_shape[i] {
                coordinates[i] = 0;
                if i <= 0 {
                    break; // Exit the loop when we reach the last dimension and wrap around
                }
                i -= 1;
            } else {
                break;
            }
        }
    }
    pub fn show(&self, prefix: String) -> std::io::Result<()> {
        let mut file = File::create(format!("{}{}{}", prefix, self.key.borrow().to_string(), ".dot"))?;
        file.write_all(b"digraph structs {\nedge [arrowhead=vee arrowsize=1]\n").expect("Unable to write to file");
        for (ix, n) in self.input.borrow().iter() {
            let neuron = n.borrow();
            let string = format!("\"{}\" [label=\"K={} V={} B={} A={}\"]\n", ix, ix, neuron.value.borrow(), neuron.bias.get_value(), neuron.activation_function.borrow());
            file.write_all(string.as_bytes()).expect("Unable to write to file");
        }
        for (ix, n) in self.neurons.borrow().iter() {
            let neuron = n.borrow();
            let string = format!("\"{}\" [label=\"K={} V={} B={} A={}\"]\n", ix, ix, neuron.value.borrow(), neuron.bias.get_value(), neuron.activation_function.borrow());
            file.write_all(string.as_bytes()).expect("Unable to write to file");
        }
        for (ix, n) in self.output.borrow().iter() {
            let neuron = n.borrow();
            let string = format!("\"{}\" [label=\"K={} V={} B={} A={}\"]\n", ix, ix, neuron.value.borrow(), neuron.bias.get_value(), neuron.activation_function.borrow());
            file.write_all(string.as_bytes()).expect("Unable to write to file");
        }

        for ((input_key, output_key), c) in self.connections.borrow().iter() {
            let string = format!("\"{}\" -> \"{}\" [label=\"W={}\"]\n", input_key, output_key, c.borrow().weight.get_value());
            file.write_all(string.as_bytes()).expect("Unable to write to file");
        }

        file.write_all(b"}\n").expect("Unable to write to file");
        Ok(())
    }
    fn create_input_neuron(&self) -> (String, RefCell<Neuron>) {
        let n = RefCell::new(Neuron::new(Type::Input, 0.5));
        let nk = n.borrow().key.clone();
        self.input.borrow().insert(nk.clone(), n.clone());
        (nk, n.clone())
    }
    fn create_neuron(&self) -> (String, RefCell<Neuron>) {
        let n = RefCell::new(Neuron::new(Type::Hidden, 0.5));
        let nk = n.borrow().key.clone();
        self.neurons.borrow().insert(nk.clone(), n.clone());
        (nk, n)
    }
    fn create_output_neuron(&self) -> (String, RefCell<Neuron>) {
        let n = RefCell::new(Neuron::new(Type::Output, 0.5));
        let nk = n.borrow().key.clone();
        self.output.borrow().insert(nk.clone(), n.clone());
        (nk, n)
    }
    fn create_connection(&self, in_key: String, in_type: Type, on_key: String, on_type: Type, weight: f32) {
        let connection = Connection::new(in_key, in_type, on_key, on_type, weight);
        self.connections.borrow().insert((connection.input_key.clone(), connection.output_key.clone()), RefCell::new(connection));
    }
    fn get_neuron(&self, ctype: &Type, key: &String) -> Result<RefCell<Neuron>, String> {
        match ctype {
            Type::Input => self.input.borrow().get(key).ok_or("Index out of bounds in Input".to_string()).clone(),
            Type::Hidden => self.neurons.borrow().get(key).ok_or("Index out of bounds in Hidden".to_string()).clone(),
            Type::Output => self.output.borrow().get(key).ok_or("Index out of bounds in Output".to_string()).clone(),
        }
    }
    fn update_ancestors(&self, parent1: String, parent2: String) {
        let mut ancestors_ref = self.ancestors.borrow_mut();
        ancestors_ref[0] = parent1;
        ancestors_ref[1] = parent2;
    }
    pub fn set_generation(&self, generation: &i32) {
        *self.generation.borrow_mut() = *generation;
    }
    pub fn activate(&self, inputs: &Vec<f32>) -> Vec<f32> {
        // Check if the length of inputs matches self.num_inputs
        // if inputs.len() != self.num_inputs as usize {
        //     panic!("Expected {} inputs, but got {}", self.num_inputs, inputs.len());
        // }
        for (_, n) in self.neurons.borrow().iter() {
            n.borrow().deactivate();
        }
        for (input_value, (_, input_neuron)) in inputs.iter().zip(self.input.borrow().iter()) {
            let input_neuron = input_neuron.borrow();
            input_neuron.set_value(*input_value);
            input_neuron.set_activated();
        }
        let mut outputs: Vec<f32> = Vec::new();
        for (key, _) in self.output.borrow().iter() {
            let res = self.activate_neuron(&Type::Output, &key);
            outputs.push(res);
        }
        outputs
    }

    fn activate_neuron(&self, ctype: &Type, key: &String) -> f32 {
        let neuron = self.get_neuron(&ctype, key).unwrap().clone();
        let mut results = vec![];
        for ((input_key, output_key), c) in self.connections.borrow().iter() {
            let connection = c.borrow();
            if *output_key == neuron.borrow().key {
                let input_neuron = self.get_neuron(&connection.input_type, &input_key).unwrap();
                if !*input_neuron.borrow().activated.borrow() {
                    self.activate_neuron(&connection.input_type, &input_key);
                }
                results.push(*input_neuron.borrow().value.borrow() * connection.weight.get_value());
            }
        }
        let x = neuron.borrow().activate(&results);
        return x
    }

    fn connection_makes_loop(&self, input_neuron_key: String, output_neuron_key: String) -> bool {
        let mut is_loop = false;
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(output_neuron_key);


        while let Some(neuron_key) = queue.pop_front() {
            if neuron_key == input_neuron_key {
                is_loop = true;
                break;
            }

            let neighbor_keys: Vec<String> = self.connections.borrow().iter()
                .filter_map(|((input_key, output_key), _)| {
                    if *input_key == neuron_key {
                        Some(output_key.clone())
                    } else {
                        None
                    }
                })
                .collect();

            for neighbor_key in neighbor_keys {
                if neighbor_key == input_neuron_key {
                    is_loop = true;
                    break;
                }
                queue.push_back(neighbor_key);
            }
        }

        is_loop
    }

    fn random_input(&self, rng_g: Option<Box<dyn RngCore>>) -> Option<(String, RefCell<Neuron>)> {
        let mut rng = rng_g.unwrap_or(Box::new(thread_rng()));
        self.input.borrow().choose(&mut rng)
    }
    fn random_hidden(&self, rng_g: Option<Box<dyn RngCore>>) -> Option<(String, RefCell<Neuron>)> {
        let mut rng = rng_g.unwrap_or(Box::new(thread_rng()));
        self.neurons.borrow().choose(&mut rng)
    }
    fn random_output(&self, rng_g: Option<Box<dyn RngCore>>) -> Option<(String, RefCell<Neuron>)> {
        let mut rng = rng_g.unwrap_or(Box::new(thread_rng()));
        self.output.borrow().choose(&mut rng)
    }
    fn mutate_add_connection(&self) {
        let (input_neuron_key, input_neuron) = self.random_input(None).unwrap();
        let (output_neuron_key, output_neuron) = self.random_output(None).unwrap();
        if input_neuron_key == output_neuron_key {
            return;
        }
        if self.connections.borrow().contains(&(input_neuron_key.clone(), output_neuron_key.clone())) {
            return;
        }
        if self.connection_makes_loop(input_neuron_key.clone(), output_neuron_key.clone()) {
            return;
        }
        self.create_connection(input_neuron_key.clone(), input_neuron.borrow().t.clone(), output_neuron_key.clone(), output_neuron.borrow().t.clone(), 0.5);
    }
    fn mutate_delete_neuron(&self) {
        let keys = self.neurons.borrow().get_keys();
        if !&keys.is_empty() {
            let del_key = keys.choose(&mut rand::thread_rng()).unwrap().clone();
            let mut connections_to_delete: Vec<(String, String)> = vec![];
            for ((input_key, output_key), _) in self.connections.borrow().iter() {
                if del_key == input_key || del_key == output_key {
                    connections_to_delete.push((input_key.clone(), output_key.clone()));
                }
            }
            for ix in connections_to_delete {
                self.connections.borrow().remove(&ix);
            }
            self.neurons.borrow().remove(&del_key);
        }
    }

    fn mutate_add_neuron(&self) {
        if self.connections.borrow().is_empty() {
            self.mutate_add_connection();
            return;
        }
        let ((input_key, output_key), c) = self.connections.borrow().choose(&mut rand::thread_rng()).unwrap();
        let connection = c.borrow();
        let (new_neuron_key, new_neuron) = self.create_neuron();
        self.create_connection(input_key.clone(), connection.input_type.clone(), new_neuron_key.clone(), new_neuron.borrow().t.clone(), connection.weight.get_value());
        self.create_connection(new_neuron_key, new_neuron.borrow().t.clone(), output_key.clone(), connection.output_type.clone(), connection.weight.get_value());
        self.connections.borrow().remove(&(input_key.clone(), output_key.clone()));
    }

    fn mutate_delete_connection(&self) {
        if self.connections.borrow().is_empty() {
            return;
        }

        let ((input_key, output_key), _) = self.connections.borrow().choose(&mut rand::thread_rng()).unwrap();

        if self.input.borrow().contains(&input_key) {
            return;
        }
        if self.output.borrow().contains(&output_key) {
            return;
        }
        self.connections.borrow().remove(&(input_key.clone(), output_key.clone()));
    }
    pub fn mutate(&self, generation: &i32) {
        self.set_generation(generation);
        if random::<f32>() < *self.neuron_add_prob.borrow() {
            self.mutate_add_neuron()
        }
        if random::<f32>() < *self.neuron_delete_prob.borrow() {
            self.mutate_delete_neuron()
        }
        if random::<f32>() < *self.conn_add_prob.borrow() {
            self.mutate_add_connection()
        }
        if random::<f32>() < *self.conn_delete_prob.borrow() {
            self.mutate_delete_connection()
        }
        for (_, connection) in self.connections.borrow().iter() {
            connection.borrow().mutate();
        }
        // Mutate hidden neurons
        for (_, neuron) in self.neurons.borrow().iter() {
            neuron.borrow().mutate();
        }
        // Mutate output neurons
        for (_, neuron) in self.output.borrow().iter() {
            neuron.borrow().mutate();
        }
    }

    pub fn crossover(&self, genome1: &Genome, genome2: &Genome) {
        let mut parent1: &Genome = &Genome::default();
        let mut parent2: &Genome = &Genome::default();
        if genome1.fitness > genome2.fitness {
            (parent1, parent2) = (genome1, genome2);
        } else {
            (parent1, parent2) = (genome2, genome1);
        }
        *self.input.borrow_mut() = parent1.input.borrow().clone();
        *self.output.borrow_mut() = parent1.output.borrow().clone();

        assert_ne!(&parent1.input.borrow() as *const _, &self.input.borrow() as *const _);
        assert_ne!(&parent1.output.borrow() as *const _, &self.output.borrow() as *const _);

        for (key, neuron1) in parent1.neurons.borrow().iter() {
            let neuron2 = parent2.neurons.borrow().get(&key);
            if neuron2.is_none() {
                self.neurons.borrow().insert(key.clone(), RefCell::new(neuron1.borrow().clone()));
            } else {
                let neuron2 = neuron2.unwrap();
                self.neurons.borrow().insert(key.clone(), neuron1.borrow().crossover(&neuron2.borrow()));
            }
        }
        for ((input_key, output_key), connection1) in parent1.connections.borrow().iter() {
            let connection2 = parent2.connections.borrow().get(&(input_key.clone(), output_key.clone()));
            if connection2.is_none() {
                self.connections.borrow().insert((input_key.clone(), output_key.clone()), connection1.clone());
            } else {
                let connection2 = connection2.unwrap();
                self.connections.borrow().insert((input_key.clone(), output_key.clone()), connection1.borrow().crossover(&connection2.borrow()));
            }
        }
        self.update_ancestors(parent1.key.borrow().clone(), parent2.key.borrow().clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_genome() {
        let genome = Genome::default();
    }
}
