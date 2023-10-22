use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::ops::Deref;
use rand::seq::SliceRandom;
use std::rc::Rc;
use rand::{random, thread_rng};
use crate::connection::Connection;
use crate::neuron::Neuron;
use serde::{Serialize, Serializer};
use crate::serde::ser::SerializeStruct;
use std::fs::File;
use std::io::prelude::*;
use rand::distributions::uniform::SampleBorrow;

const NEURON_ADD_PROB: f32 = 0.01;
const NEURON_DELETE_PROB: f32 = 0.01;
const CONN_ADD_PROB: f32 = 0.1;
const CONN_DELETE_PROB: f32 = 0.1;
// const NEURON_ADD_PROB: f32 = 0.05;
// const NEURON_DELETE_PROB: f32 = 0.01;
// const CONN_ADD_PROB: f32 = 0.05;
// const CONN_DELETE_PROB: f32 = 0.01;

#[derive(Debug)]
pub struct Genome {
    pub key: u32,
    pub num_inputs: i32,
    pub num_outputs: i32,
    pub neuron_add_prob: RefCell<f32>,
    pub neuron_delete_prob: RefCell<f32>,
    pub conn_add_prob: RefCell<f32>,
    pub conn_delete_prob: RefCell<f32>,
    adjusted_fitness: f32,
    compatibility_disjoint_coefficient: f32,
    pub generation: RefCell<i32>,
    ancestors: RefCell<[u32; 2]>,
    neurons: RefCell<HashMap<i32, Rc<Neuron>>>,
    connections: RefCell<HashMap<(i32, i32), Rc<Connection>>>,
    pub fitness: RefCell<f32>,
}

impl Serialize for Genome {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
    {
        let mut state = serializer.serialize_struct("Genome", 15)?; // 15 is the number of fields in Genome

        state.serialize_field("key", &self.key)?;
        state.serialize_field("num_inputs", &self.num_inputs)?;
        state.serialize_field("num_outputs", &self.num_outputs)?;
        state.serialize_field("neuron_add_prob", &*self.neuron_add_prob.borrow())?;
        state.serialize_field("neuron_delete_prob", &*self.neuron_delete_prob.borrow())?;
        state.serialize_field("conn_add_prob", &*self.conn_add_prob.borrow())?;
        state.serialize_field("conn_delete_prob", &*self.conn_delete_prob.borrow())?;
        state.serialize_field("adjusted_fitness", &self.adjusted_fitness)?;
        state.serialize_field("compatibility_disjoint_coefficient", &self.compatibility_disjoint_coefficient)?;
        state.serialize_field("generation", &self.generation)?;
        state.serialize_field("ancestors", &*self.ancestors.borrow())?;
        state.serialize_field("neurons", &*self.neurons.borrow())?;
        state.serialize_field("connections", &self.connections.borrow().iter().map(|((input, output), connection)| {
            (format!("({}, {})", input, output), Rc::clone(connection))
        }).collect::<HashMap<String, Rc<Connection>>>())?;
        state.serialize_field("fitness", &*self.fitness.borrow())?;

        state.end()
    }
}

impl Default for Genome {
    fn default() -> Self {
        Genome {
            key: 0,
            num_inputs: 0,
            num_outputs: 0,
            neuron_add_prob: RefCell::new(NEURON_ADD_PROB),
            neuron_delete_prob: RefCell::new(NEURON_DELETE_PROB),
            conn_add_prob: RefCell::new(CONN_ADD_PROB),
            conn_delete_prob: RefCell::new(CONN_DELETE_PROB),
            adjusted_fitness: 0.0,
            compatibility_disjoint_coefficient: 0.0,
            generation: RefCell::new(0),
            ancestors: RefCell::new([0,0]),
            neurons: RefCell::new(HashMap::new()),
            connections: RefCell::new(HashMap::new()),
            fitness: RefCell::new(0.0),
        }
    }
}

impl Genome {
    pub fn new(key: u32, num_inputs: i32, num_outputs: i32, _size: i32) -> Self {
        let _rng = thread_rng();
        let g = Genome {
            key,
            num_inputs,
            num_outputs,
            // neuron_add_prob: RefCell::new(rng.gen()),
            // neuron_delete_prob: RefCell::new(rng.gen()),
            // conn_add_prob: RefCell::new(rng.gen()),
            // conn_delete_prob: RefCell::new(rng.gen()),
            ..Genome::default()
        };
        for i in 0..num_inputs {
            g.create_neuron(-i - 1, false);
        }
        for i in 0..num_outputs {
            g.create_neuron(i + 1, true);
        }
        let input_neurons = g.get_input_neuron_keys();
        let output_neurons = g.get_output_neuron_keys();

        for &input_neuron in &input_neurons {
            for &output_neuron in &output_neurons {
                g.create_connection(input_neuron, output_neuron, 0.0);
            }
        }
        g
    }

    pub fn show(&self, prefix: String) -> std::io::Result<()> {
        let mut file = File::create(format!("{}{}{}", prefix, self.key.to_string(), ".dot"))?;
        file.write_all(b"digraph structs {\nedge [arrowhead=vee arrowsize=1]\n").expect("Unable to write to file");
        for n in self.neurons.borrow_mut().values() {
            let string = format!("{} [label=\"K={} V={} B={} A={}\"]\n", n.key, n.key, n.value.borrow(), n.bias.get_value(), n.activation_function.borrow());
            file.write_all(string.as_bytes()).expect("Unable to write to file");
        }

        for c in self.connections.borrow().values() {
            let string = format!("{} -> {} [label=\"W={}\"]\n", c.input_key, c.output_key, c.weight.get_value());
            file.write_all(string.as_bytes()).expect("Unable to write to file");
        }

        file.write_all(b"}\n").expect("Unable to write to file");
        Ok(())
    }

    fn get_new_neuron_key(&self) -> i32 {
        let neurones = self.neurons.borrow();
        neurones.keys().max().map_or(1, |&max_key| max_key + 1)
    }

    fn create_neuron(&self, key: i32, output: bool) -> Rc<Neuron> {
        let n = Rc::new(Neuron::new(key, output, 0.0));
        self.neurons.borrow_mut().insert(key, Rc::clone(&n));
        Rc::clone(&n)
    }
    fn create_connection(&self, input_neurone_key: i32, output_neurone_key: i32, weight: f32) {
        let connection = Connection::new(input_neurone_key, output_neurone_key, weight);
        self.connections.borrow_mut().insert(connection.get_key(), Rc::new(connection));
    }

    fn get_input_neuron_keys(&self) -> Vec<i32> {
        self.neurons.borrow().iter()
            .filter_map(|(&key, _)| {
                if key < 0 {
                    Some(key)
                } else {
                    None
                }
            })
            .collect()
    }
    fn get_hidden_neuron_keys(&self) -> Vec<i32> {
        self.neurons.borrow().iter()
            .filter_map(|(&key, n)| {
                if !n.output && key > 0 {
                    Some(key)
                } else {
                    None
                }
            })
            .collect()
    }
    fn get_output_neuron_keys(&self) -> Vec<i32> {
        self.neurons.borrow().iter()
            .filter_map(|(&key, n)| {
                if n.output {
                    Some(key)
                } else {
                    None
                }
            })
            .collect()
    }

    fn get_neurone(&self, key: &i32) -> Rc<Neuron> {
        self.neurons.borrow().get(key).cloned().unwrap_or_else(|| {
            panic!("Neuron with key {} not found!", key)
        })
    }
    fn update_ancestors(&self, parent1: u32, parent2: u32) {
        let mut ancestors_ref = self.ancestors.borrow_mut();
        ancestors_ref[0] = parent1;
        ancestors_ref[1] = parent2;
    }
    pub fn set_generation(&self, generation: &i32) {
        *self.generation.borrow_mut() = *generation;
    }

    pub fn activate(&self, inputs: &Vec<f32>) -> Vec<f32> {
        // Check if the length of inputs matches self.num_inputs
        if inputs.len() != self.num_inputs as usize {
            panic!("Expected {} inputs, but got {}", self.num_inputs, inputs.len());
        }
        for n in self.neurons.borrow_mut().values_mut() {
            n.deactivate();
        }
        for (input_value, input_neuron_key) in inputs.iter().zip(self.get_input_neuron_keys()) {
            let input_neuron = self.get_neurone(&input_neuron_key);
            input_neuron.set_value(*input_value);
            *input_neuron.activated.borrow_mut() = true;
        }
        let mut outputs : Vec<f32> = Vec::new();
        for key in self.get_output_neuron_keys() {
            outputs.push(self.activate_neurone(&key));
        }
        outputs
    }

    fn activate_neurone(&self, key: &i32) -> f32 {
        let neurone = self.get_neurone(key);
        let mut results = vec![];
        for c in self.connections.borrow().values() {
            if c.output_key == neurone.key {
                let input_neurone = self.get_neurone(&c.input_key);
                if !*input_neurone.activated.borrow() {
                    self.activate_neurone(&input_neurone.key);
                }
                results.push(*input_neurone.value.borrow() * c.weight.get_value());
            }
        }
        neurone.activate(&results)
    }

    fn connection_makes_loop(&self, input_neuron_key: i32, output_neuron_key: i32) -> bool {
        let mut is_loop = false;
        let mut queue = VecDeque::new();
        queue.push_back(output_neuron_key);

        let connections = self.connections.borrow();
        let neurons = self.neurons.borrow();

        while let Some(neuron_key) = queue.pop_front() {
            if neuron_key == input_neuron_key {
                is_loop = true;
                break;
            }

            let neighbor_keys: Vec<i32> = connections
                .iter()
                .filter_map(|(&(input_key, output_key), _)| {
                    if input_key == neuron_key {
                        Some(output_key)
                    } else {
                        None
                    }
                })
                .collect();

            for &neighbor_key in &neighbor_keys {
                if !neurons.contains_key(&neighbor_key) {
                    continue;
                }
                if neighbor_key == input_neuron_key {
                    is_loop = true;
                    break;
                }
                queue.push_back(neighbor_key);
            }
        }

        is_loop
    }

    fn mutate_add_connection(&self) {
        let possible_inputs: Vec<_> = [&self.get_input_neuron_keys()[..], &self.get_hidden_neuron_keys()[..]].concat();
        let input_neurone_key = *possible_inputs.choose(&mut rand::thread_rng()).unwrap();
        let possible_outputs: Vec<_> = [&self.get_hidden_neuron_keys()[..], &self.get_output_neuron_keys()[..]].concat();
        let output_neurone_key = *possible_outputs.choose(&mut rand::thread_rng()).unwrap();

        if input_neurone_key == output_neurone_key {
            return;
        }
        {
            let connections = self.connections.borrow(); // Borrowing the RefCell
            if connections.contains_key(&(input_neurone_key, output_neurone_key))
                || connections.contains_key(&(output_neurone_key, input_neurone_key))
            {
                return;
            }
        }
        if self.connection_makes_loop(input_neurone_key, output_neurone_key) {
            return;
        }
        self.create_connection(input_neurone_key, output_neurone_key, random());
    }
    fn mutate_delete_neurone(&self) {
        let keys = self.get_hidden_neuron_keys();
        if !&keys.is_empty() {
            let del_key = *keys.choose(&mut rand::thread_rng()).unwrap();
            let mut connections_to_delete : Vec<(i32, i32)> = vec![];
            {
                for ck in self.connections.borrow().keys() {
                    if del_key == ck.0 || del_key == ck.1 {
                        connections_to_delete.push(*ck);
                    }
                }
            }
            {
                let mut connections = self.connections.borrow_mut();
                for ck in connections_to_delete {
                    connections.remove(&ck);
                }
            }
            self.neurons.borrow_mut().remove(&del_key);
        }
    }

    fn mutate_add_neurone(&self) {
        if self.connections.borrow().is_empty() {
            self.mutate_add_connection();
            return;
        }
        let connection_keys : Vec<Rc<Connection>> = self.connections.borrow().values().cloned().collect();
        if let Some(connection_to_split) = connection_keys.choose(&mut rand::thread_rng()) {
            let new_neuron = self.create_neuron(self.get_new_neuron_key(), false);
            self.create_connection(connection_to_split.input_key, new_neuron.key, connection_to_split.weight.get_value());
            self.create_connection(new_neuron.key, connection_to_split.output_key, connection_to_split.weight.get_value());
            self.connections.borrow_mut().remove(&connection_to_split.get_key());
        } else {
            return;
        }
    }

    fn mutate_delete_connection(&self) {
        if self.connections.borrow().is_empty() {
            return;
        }

        let connection_to_del = {
            let connections = self.connections.borrow();
            let connections_vec: Vec<Rc<Connection>> = connections.values().cloned().collect();
            connections_vec.choose(&mut rand::thread_rng()).cloned()
        };

        if let Some(connection) = connection_to_del {
            if self.get_input_neuron_keys().contains(&connection.input_key) {
                return;
            }
            if self.get_output_neuron_keys().contains(&connection.output_key) {
                return;
            }
            self.connections.borrow_mut().remove(&connection.get_key());
        }
    }
    pub fn mutate(&self) {
        if random::<f32>() < *self.neuron_add_prob.borrow() {
            self.mutate_add_neurone()
        }
        if random::<f32>() < *self.neuron_delete_prob.borrow() {
            self.mutate_delete_neurone()
        }
        if random::<f32>() < *self.conn_add_prob.borrow() {
            self.mutate_add_connection()
        }
        if random::<f32>() < *self.conn_delete_prob.borrow() {
            self.mutate_delete_connection()
        }
        for connection in self.connections.borrow().values() {
            connection.mutate();
        }
        for neuron in self.neurons.borrow().values() {
            neuron.mutate();
        }
    }

    pub fn crossover(&self, genome1 : &Genome, genome2: &Genome) {
        let mut parent1 : &Genome = &Genome::default();
        let mut parent2 : &Genome = &Genome::default();
        if genome1.fitness > genome2.fitness {
            (parent1, parent2) = (genome1, genome2);
        } else {
            (parent1, parent2) = (genome2, genome1);
        }
        for key in parent1.get_hidden_neuron_keys() {
            let neurons = parent1.neurons.borrow();
            let neurone1 = neurons.get(&key);
            let neurone2 = neurons.get(&key);
            // TODO: Write this simpler by using the rules we have given output neurons keys like negative numbers for checks

            assert!(!self.get_hidden_neuron_keys().contains(&key), "Key present in self hidden neurons.");
            if neurone2.is_none() {
                self.neurons.borrow_mut().insert(key, Rc::new(neurone1.unwrap().deref().clone()));
            } else {
                self.neurons.borrow_mut().insert(key, Rc::new(neurone1.unwrap().crossover(neurone2.unwrap())));
            }
        }
        for (key, connection) in parent1.connections.borrow().iter() {
            let connections = parent2.connections.borrow();
            let connection2 = connections.get(&key);
            if connection2.is_none() {
                self.connections.borrow_mut().insert(*key, Rc::new(connection.deref().clone()));
            } else {
                self.connections.borrow_mut().insert(*key, Rc::new(connection.crossover(connection2.unwrap())));
            }
        }
        self.update_ancestors(parent1.key, parent2.key);
    }
}
