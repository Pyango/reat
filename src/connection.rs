use crate::attribute::Attribute;
use rand::Rng;
use serde::{Serialize, Serializer};
use crate::serde::ser::SerializeStruct;
use bincode::{Decode, Encode};

#[derive(Encode, Decode, PartialEq, Default, Debug, Clone)]
pub struct Connection {
    compatibility_weight_coefficient: f32,
    pub input_key: i32,
    pub output_key: i32,
    pub weight: Attribute,
}

impl Serialize for Connection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
    {
        let mut state = serializer.serialize_struct("Connection", 4)?;
        state.serialize_field("compatibility_weight_coefficient", &self.compatibility_weight_coefficient)?;
        state.serialize_field("input_key", &self.input_key)?;
        state.serialize_field("output_key", &self.output_key)?;
        state.serialize_field("weight", &self.weight)?;
        state.end()
    }
}

impl Connection {
    pub fn new(input_key: i32, output_key: i32, weight: f32) -> Self {
        let n = Connection {
            input_key,
            output_key,
            weight: Attribute::new(weight),
            ..Connection::default()
        };
        n
    }

    pub fn get_key(&self) -> (i32, i32) {
        (self.input_key, self.output_key)
    }

    pub fn distance(&self, other: &Connection) -> f32 {
        let d = (self.weight.get_value() - other.weight.get_value()).abs();
        d * self.compatibility_weight_coefficient
    }

    pub fn copy(&self) -> Connection {
        Connection::new(self.input_key, self.output_key, self.weight.get_value())
    }

    pub fn crossover(&self, connection: &Connection) -> Connection {
        let weight_value = if rand::thread_rng().gen::<f32>() > 0.5 {
            connection.weight.get_value()
        } else {
            self.weight.get_value()
        };
        Connection::new(self.input_key, self.output_key, weight_value)
    }

    pub fn mutate(&self) -> f32 {
        self.weight.mutate_value(None)
    }
}
