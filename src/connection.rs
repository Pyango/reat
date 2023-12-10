use std::cell::RefCell;
use crate::attribute::Attribute;
use rand::Rng;
use bincode::{Decode, Encode};
use crate::clone::CustomClone;
use crate::t::Type;

#[derive(Encode, Decode, PartialEq, Default, Debug, Clone)]
pub struct Connection {
    compatibility_weight_coefficient: f32,
    pub input_key: String,
    pub input_type: Type,
    pub output_key: String,
    pub output_type: Type,
    pub weight: Attribute,
}

impl CustomClone for Connection {
    fn clone(&self) -> Self {
        Connection {
            compatibility_weight_coefficient: self.compatibility_weight_coefficient,
            input_key: self.input_key.clone(),
            input_type: self.input_type.clone(),
            output_key: self.output_key.clone(),
            output_type: self.output_type.clone(),
            weight: self.weight.clone(),
        }
    }
}

impl Connection {
    pub fn new(input_key: String, input_type: Type, output_key: String, output_type: Type, weight: f32) -> Self {
        let n = Connection {
            input_key,
            input_type,
            output_key,
            output_type,
            weight: Attribute::new(weight),
            ..Connection::default()
        };
        n
    }
    pub fn get_key(&self) -> (&str, &str) {
        (&self.input_key, &self.output_key)
    }
    pub fn distance(&self, other: &Connection) -> f32 {
        let d = (self.weight.get_value() - other.weight.get_value()).abs();
        d * self.compatibility_weight_coefficient
    }

    pub fn crossover(&self, connection: &Connection) -> RefCell<Connection> {
        let weight_value = if rand::thread_rng().gen::<f32>() > 0.5 {
            connection.weight.get_value()
        } else {
            self.weight.get_value()
        };
        RefCell::new(Connection::new(self.input_key.clone(), self.input_type.clone(), self.output_key.clone(), self.output_type.clone(), weight_value))
    }

    pub fn mutate(&self) -> f32 {
        self.weight.mutate_value(None)
    }
}
