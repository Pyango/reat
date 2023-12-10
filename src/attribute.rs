use std::cell::{RefCell};
use std::rc::Rc;
use bincode::{Decode, Encode};
use rand::{Rng, RngCore, thread_rng};
use rand_distr::{Distribution, Normal};


const MAX_VALUE: f32 = 30.0;
const MIN_VALUE: f32 = -30.0;
const MUTATE_RATE: f32 = 0.05;
const MUTATE_POWER: f32 = 0.05;
const REPLACE_RATE: f32 = 0.05;

#[derive(Encode, Decode, PartialEq, Debug, Clone)]
pub struct Attribute {
    value: Rc<RefCell<f32>>,
    initial_value: Rc<RefCell<f32>>,
    max_value: f32,
    min_value: f32,
    mutate_rate: f32,
    mutate_power: f32,
    replace_rate: f32,
}

impl Default for Attribute {
    fn default() -> Self {
        Attribute {
            value: Rc::new(RefCell::new(0.0)),
            initial_value: Rc::new(RefCell::new(0.0)),
            max_value: MAX_VALUE,
            min_value: MIN_VALUE,
            mutate_rate: MUTATE_RATE,
            mutate_power: MUTATE_POWER,
            replace_rate: REPLACE_RATE,
        }
    }
}

impl Attribute {
    pub fn new(value: f32) -> Self {
        Attribute {
            value: Rc::new(RefCell::new(value)),
            initial_value: Rc::new(RefCell::new(value)),
            max_value: MAX_VALUE,
            min_value: MIN_VALUE,
            mutate_rate: MUTATE_RATE,
            mutate_power: MUTATE_POWER,
            replace_rate: REPLACE_RATE,
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

    pub fn mutate_value(&self, rng_g: Option<Box<dyn RngCore>>) -> f32 {
        let mut rng = rng_g.unwrap_or(Box::new(thread_rng()));
        let r: f32 = rng.gen();

        if r < self.mutate_rate {
            let normal = Normal::new(0.0, self.mutate_power as f64)
                .expect("Failed to create normal distribution");
            let mutation: f32 = normal.sample(&mut *rng) as f32;
            let new_value = self.clamp(*self.value.clone().borrow() + mutation);
            *self.value.borrow_mut() = new_value;
            return new_value;
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
    fn test_mutate_positive_function() {
        let mut my_rng = SimpleFloatRng::new(0.7, 0.01);
        let mut attr = Attribute {
            value: Rc::new(RefCell::new(1.0)),
            initial_value: Rc::new(RefCell::new(1.0)),
            max_value: 30.0,
            min_value: -30.0,
            mutate_rate: 1.0,
            mutate_power: 0.1,
            replace_rate: 0.01,
        };
        assert_relative_eq!(attr.mutate_value(Some(Box::new(my_rng))), 1.1720734);
    }

    #[test]
    fn test_mutate_negative_function() {
        let mut my_rng = SimpleFloatRng::new(0.2, 0.01);
        let mut attr = Attribute {
            value: Rc::new(RefCell::new(1.0)),
            initial_value: Rc::new(RefCell::new(1.0)),
            max_value: 30.0,
            min_value: -30.0,
            mutate_rate: 1.0,
            mutate_power: 0.1,
            replace_rate: 0.01,
        };
        assert_relative_eq!(attr.mutate_value(Some(Box::new(my_rng))), 0.7809976);
    }

    #[test]
    fn test_replace_function() {
        let mut my_rng = SimpleFloatRng::new(0.2, 0.01);
        let mut attr = Attribute {
            value: Rc::new(RefCell::new(0.4245673)),
            initial_value: Rc::new(RefCell::new(1.0)),
            max_value: 30.0,
            min_value: -30.0,
            mutate_rate: 0.0,
            mutate_power: 0.1,
            replace_rate: 0.5,
        };
        assert_relative_eq!(attr.mutate_value(Some(Box::new(my_rng))), 1.0);
    }
}
