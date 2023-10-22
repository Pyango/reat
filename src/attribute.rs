use std::cell::{RefCell};


use rand::{Rng, RngCore, thread_rng};
use rand::rngs::ThreadRng;
use rand_distr::{Distribution, Normal};
use serde::{Serialize, Serializer};

use crate::serde::ser::SerializeStruct;

const MAX_VALUE: f32 = 30.0;
const MIN_VALUE: f32 = -30.0;
const MUTATE_RATE: f32 = 0.05;
const MUTATE_POWER: f32 = 0.05;
const REPLACE_RATE: f32 = 0.05;

struct SimpleFloatRng {
    value: f32,
    step: f32,
}

impl SimpleFloatRng {
    fn new(start: f32, step: f32) -> Self {
        SimpleFloatRng { value: start, step }
    }

    fn gen(&mut self) -> f32 {
        self.value += self.step;
        if self.value >= 1.0 {
            self.value -= 1.0;
        }
        self.value
    }
}

impl RngCore for SimpleFloatRng {
    fn next_u32(&mut self) -> u32 {
        (self.gen() * u32::MAX as f32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        (self.gen() * u64::MAX as f32) as u64
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for byte in dest.iter_mut() {
            *byte = (self.gen() * 255.0) as u8;
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Attribute<R: Rng = ThreadRng> {
    value: RefCell<f32>,
    initial_value: RefCell<f32>,
    max_value: f32,
    min_value: f32,
    mutate_rate: f32,
    mutate_power: f32,
    replace_rate: f32,
    rng: RefCell<R>,
}

impl Default for Attribute<ThreadRng> {
    fn default() -> Self {
        Attribute {
            value: RefCell::new(0.0),
            initial_value: RefCell::new(0.0),
            max_value: MAX_VALUE,
            min_value: MIN_VALUE,
            mutate_rate: MUTATE_RATE,
            mutate_power: MUTATE_POWER,
            replace_rate: REPLACE_RATE,
            rng: RefCell::new(thread_rng()),
        }
    }
}

impl Serialize for Attribute<ThreadRng> {
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

impl Attribute<ThreadRng> {
    pub fn new(value: f32) -> Self {
        Attribute {
            value: RefCell::new(value),
            initial_value: RefCell::new(value),
            rng: RefCell::new(thread_rng()),
            max_value: MAX_VALUE,
            min_value: MIN_VALUE,
            mutate_rate: MUTATE_RATE,
            mutate_power: MUTATE_POWER,
            replace_rate: REPLACE_RATE,
        }
    }
}

impl<R: Rng> Attribute<R> {

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
        let mut rng = self.rng.borrow_mut();
        let r: f32 = rng.gen();

        if r < self.mutate_rate {
            let normal = Normal::new(0.0, self.mutate_power as f64)
                .expect("Failed to create normal distribution");
            let mutation: f32 = normal.sample(&mut *rng) as f32;
            *self.value.borrow_mut() = self.clamp(*self.value.clone().borrow() + mutation);
            return *self.value.borrow();
        }

        if r < self.replace_rate + self.mutate_rate {
            *self.value.borrow_mut() = *self.initial_value.borrow();
            return *self.value.borrow();
        }

        *self.value.borrow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_custom_simple_float_rng() {
        let mut my_rng = SimpleFloatRng::new(0.0, 0.1);
        let sample: f32 = my_rng.gen();
        assert_eq!(sample, 0.1);
        let sample: f32 = my_rng.gen();
        assert_eq!(sample, 0.2);
    }

    #[test]
    fn test_mutate_positive_function() {
        let mut my_rng = SimpleFloatRng::new(0.7, 0.01);
        let mut attr: Attribute<SimpleFloatRng> = Attribute {
            value: RefCell::new(1.0),
            initial_value: RefCell::new(1.0),
            max_value: 30.0,
            min_value: -30.0,
            mutate_rate: 1.0,
            mutate_power: 0.1,
            replace_rate: 0.01,
            rng: RefCell::new(my_rng),
        };
        assert_relative_eq!(attr.mutate_value(), 1.1720734);
    }

    #[test]
    fn test_mutate_negative_function() {
        let mut my_rng = SimpleFloatRng::new(0.2, 0.01);
        let mut attr: Attribute<SimpleFloatRng> = Attribute {
            value: RefCell::new(1.0),
            initial_value: RefCell::new(1.0),
            max_value: 30.0,
            min_value: -30.0,
            mutate_rate: 1.0,
            mutate_power: 0.1,
            replace_rate: 0.01,
            rng: RefCell::new(my_rng),
        };
        assert_relative_eq!(attr.mutate_value(), 0.7809976);
    }

    #[test]
    fn test_replace_function() {
        let mut my_rng = SimpleFloatRng::new(0.2, 0.01);
        let mut attr: Attribute<SimpleFloatRng> = Attribute {
            value: RefCell::new(0.4245673),
            initial_value: RefCell::new(1.0),
            max_value: 30.0,
            min_value: -30.0,
            mutate_rate: 0.0,
            mutate_power: 0.1,
            replace_rate: 0.5,
            rng: RefCell::new(my_rng),
        };
        assert_relative_eq!(attr.mutate_value(), 1.0);
    }

    #[test]
    fn test_gauss_function() {
        let mut my_rng = SimpleFloatRng::new(0.7, 0.01);
        let normal = Normal::new(0.0, 0.1).expect("Failed to create normal distribution");
        let r: f32 = my_rng.gen();
        let mutation: f32 = normal.sample(&mut my_rng);
        assert_relative_eq!(mutation, 0.1720734);
    }
}
