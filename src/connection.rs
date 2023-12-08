use std::cell::RefCell;
use crate::attribute::Attribute;
use rand::Rng;
use bincode::{Decode, Encode};

#[derive(Encode, Decode, PartialEq, Default, Debug, Clone)]
pub enum ConnectionType {
    #[default]
    Input,
    Output,
    Hidden,
}

#[derive(Encode, Decode, PartialEq, Default, Debug, Clone)]
pub struct Connection {
    compatibility_weight_coefficient: f32,
    pub c_type: ConnectionType,
    pub input_key: String,
    pub output_key: String,
    pub weight: Attribute,
}

impl Connection {
    pub fn new(c_type: ConnectionType, input_key: String, output_key: String, weight: f32) -> Self {
        let n = Connection {
            c_type,
            input_key,
            output_key,
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

    pub fn copy(&self) -> Connection {
        Connection::new(self.c_type.clone(), self.input_key.clone(), self.output_key.clone(), self.weight.get_value())
    }

    pub fn crossover(&self, connection: &Connection) -> RefCell<Connection> {
        let weight_value = if rand::thread_rng().gen::<f32>() > 0.5 {
            connection.weight.get_value()
        } else {
            self.weight.get_value()
        };
        RefCell::new(Connection::new(self.c_type.clone(), self.input_key.clone(), self.output_key.clone(), weight_value))
    }

    pub fn mutate(&self) -> f32 {
        self.weight.mutate_value(None)
    }
}
