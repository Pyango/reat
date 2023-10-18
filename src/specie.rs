use std::collections::HashSet;

#[derive(Default, Debug)]
pub struct Specie {
    pub key: String,
    pub genomes: HashSet<i32>,
}
