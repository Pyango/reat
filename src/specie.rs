use std::collections::HashSet;
use bincode::{config, Decode, Encode};

#[derive(Encode, Decode, PartialEq, Default, Debug)]
pub struct Specie {
    pub key: String,
    pub genomes: HashSet<i32>,
}
