use rand::{Rng, RngCore};
use rand_distr::{Distribution, Normal};

#[derive(Debug)]
pub struct SimpleFloatRng {
    value: f32,
    step: f32,
}

impl SimpleFloatRng {
    pub fn new(start: f32, step: f32) -> Self {
        SimpleFloatRng { value: start, step }
    }

    pub fn gen(&mut self) -> f32 {
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
    fn test_gauss_function() {
        let mut my_rng = SimpleFloatRng::new(0.7, 0.01);
        let normal = Normal::new(0.0, 0.1).expect("Failed to create normal distribution");
        let r: f32 = my_rng.gen();
        let mutation: f32 = normal.sample(&mut my_rng);
        assert_relative_eq!(mutation, 0.1720734);
    }
}
