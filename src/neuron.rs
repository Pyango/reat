use std::cell::RefCell;
use std::rc::Rc;
use rand::{random, Rng};
use crate::activations::FUNCTIONS;
use crate::attribute::Attribute;
use bincode::{Decode, Encode};
use crate::helpers::generate_uuid_key;
use crate::t::Type;

const ACTIVATION_FUNCTION_MUTATE_RATE: f32 = 0.01;

#[derive(Encode, Decode, PartialEq, Debug, Clone)]
pub struct Neuron {
    pub key: String,
    pub value: Rc<RefCell<f32>>,
    pub bias: Attribute,
    pub activated: Rc<RefCell<bool>>,
    pub t: Type,
    pub activation_function_mutate_rate: Rc<RefCell<f32>>,
    pub activation_function: Rc<RefCell<usize>>,
}

impl Default for Neuron {
    fn default() -> Self {
        Neuron {
            key: generate_uuid_key(),
            value: Rc::new(RefCell::new(0.0)),
            bias: Attribute::default(),
            activated: Rc::new(RefCell::new(false)),
            t: Type::Input,
            activation_function_mutate_rate: Rc::new(RefCell::new(ACTIVATION_FUNCTION_MUTATE_RATE)),
            activation_function: Rc::new(RefCell::new(0)),
        }
    }
}

impl Neuron {
    pub fn new(t: Type, bias: f32) -> Self {
        let n = Neuron {
            t,
            bias: Attribute::new(bias),
            ..Neuron::default()
        };
        n
    }

    pub fn get_value(&self) -> f32 {
        *self.value.borrow()
    }
    pub fn set_activated(&self) {
        *self.activated.borrow_mut() = true;
    }
    pub fn set_value(&self, value: f32) {
        *self.value.borrow_mut() = value;
    }
    pub fn deactivate(&self) {
        *self.activated.borrow_mut() = false;
    }
    pub fn activate(&self, input: &Vec<f32>) -> f32 {
        let sum: f32 = input.iter().sum();
        self.set_value(FUNCTIONS[*self.activation_function.borrow()](sum + self.bias.get_value()));
        *self.activated.borrow_mut() = true;
        self.get_value()
    }

    pub fn mutate(&self) {
        self.bias.mutate_value(None);
        let r: f32 = random(); // Generates a float between 0.0 and 1.0
        if r < *self.activation_function_mutate_rate.borrow() {
            *self.activation_function.borrow_mut() = rand::thread_rng().gen_range(0..=17); // Generates a integer between 0 and 17 inclusive
        }
    }

    pub fn crossover(&self, neurone1: &Neuron) -> RefCell<Neuron> {
        let mut rng = rand::thread_rng();

        RefCell::new(Neuron::new(
            Type::Hidden,
            if rng.gen::<f64>() > 0.5 { neurone1.bias.get_value() } else { self.bias.get_value() },
        ))
    }
}
