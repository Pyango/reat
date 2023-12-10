use bincode::{Decode, Encode};

#[derive(Encode, Decode, PartialEq, Default, Debug, Clone)]
pub enum Type {
    #[default]
    Input,
    Output,
    Hidden,
}
