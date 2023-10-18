use std::cell::RefCell;
use rand::{Rng, thread_rng};
use rand_distr::{Distribution, Normal};
use serde::{Serialize, Serializer};
use crate::serde::ser::SerializeStruct;

#[derive(Debug, Clone)]
pub struct Attribute {
    value: RefCell<f32>,
    initial_value: RefCell<f32>,
    max_value: f32,
    min_value: f32,
    mutate_rate: f32,
    mutate_power: f32,
    replace_rate: f32,
}

impl Default for Attribute {
    fn default() -> Self {
        Attribute {
            value: RefCell::new(0.0),
            initial_value: RefCell::new(0.0),
            max_value: 0.0,
            min_value: 0.0,
            mutate_rate: 0.1,
            mutate_power: 0.5,
            replace_rate: 0.1,
        }
    }
}


impl Serialize for Attribute {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
    {
        let mut state = serializer.serialize_struct("Attribute", 7)?;
        state.serialize_field("value", &*self.value.borrow())?;
        state.serialize_field("initial_value", &*self.initial_value.borrow())?;
        state.serialize_field("max_value", &self.max_value)?;
        state.serialize_field("min_value", &self.min_value)?;
        state.serialize_field("mutate_rate", &self.mutate_rate)?;
        state.serialize_field("mutate_power", &self.mutate_power)?;
        state.serialize_field("replace_rate", &self.replace_rate)?;
        state.end()
    }
}

impl Attribute {
    pub fn new(value: f32) -> Self {
        Attribute {
            value: RefCell::new(value),
            initial_value: RefCell::new(value),
            ..Attribute::default()
        }
    }
    pub fn get_value(&self) -> f32 {
        *self.value.borrow()
    }

    fn clamp(&self, value: f32) -> f32 {
        value.max(self.min_value).min(self.max_value)
    }
    pub fn truncate(&self, decimals: u32) -> f32 {
        if decimals == 0 {
            return self.value.borrow().trunc();
        }
        let factor = 10f32.powi(decimals as i32);
        (*self.value.borrow() * factor).trunc() / factor
    }

    pub fn mutate_value(&self) -> f32 {
        let r: f32 = thread_rng().gen();
        if r < self.mutate_rate {
            println!("mutate attribute");
            let normal = Normal::new(0.0, self.mutate_power as f64).expect("Failed to create normal distribution");
            let mutation: f32 = normal.sample(&mut thread_rng()) as f32;
            let old_value = *self.value.borrow();
            *self.value.borrow_mut() = self.clamp(old_value + mutation);
        } else if r < self.replace_rate + self.mutate_rate {
            println!("replace attribute");
            *self.value.borrow_mut() = *self.initial_value.borrow();
        }
        *self.value.borrow()
    }
}
