use std::cell::{RefCell};
use std::cmp::PartialEq;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use bincode::{Decode, Encode};
use rand::Rng;

#[derive(Encode, Decode, PartialEq, Debug, Clone)]
pub struct OrderedRefCell<K: Clone + Eq + std::hash::Hash + 'static, T: Clone> {
    map: RefCell<HashMap<K, RefCell<T>>>,
    keys: RefCell<Vec<K>>,
}

impl<K: Clone + Eq + std::hash::Hash, T: Clone> OrderedRefCell<K, T> {
    pub(crate) fn new() -> Self {
        OrderedRefCell {
            map: RefCell::new(HashMap::new()),
            keys: RefCell::new(Vec::new()),
        }
    }
    pub fn deep_clone(&self) -> Self {
        let mut new_map = HashMap::new();
        for (key, value) in self.map.borrow().iter() {
            new_map.insert(key.clone(), value.clone());
        }

        OrderedRefCell {
            map: RefCell::new(new_map),
            keys: RefCell::new(self.keys.borrow().clone()),
        }
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.keys.borrow().is_empty()
    }
    pub(crate) fn choose<R: Rng>(&self, rng: &mut R) -> Option<(K, RefCell<T>)> {
        if self.keys.borrow().is_empty() {
            None
        } else {
            let index = rng.gen_range(0..self.keys.borrow().len());
            let key = self.keys.borrow()[index].clone();
            self.map.borrow().get(&key).cloned().map(|value| (key, value))
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.map.borrow().len()
    }
    pub(crate) fn insert(&self, key: K, value: RefCell<T>) -> Option<RefCell<T>> {
        let mut map = self.map.borrow_mut();
        let mut keys = self.keys.borrow_mut();
        match map.entry(key.clone()) {
            Entry::Vacant(entry) => {
                keys.push(key);
                entry.insert(value);
                None
            }
            Entry::Occupied(mut entry) => Some(entry.insert(value)),
        }
    }
    pub(crate) fn get_keys(&self) -> Vec<K> {
        self.keys.borrow().clone()
    }
    pub(crate) fn get(&self, key: &K) -> Option<RefCell<T>> {
        self.map.borrow().get(key).cloned()
    }
    pub(crate) fn get_by_index(&self, ix: &usize) -> Option<RefCell<T>> {
        let map = self.map.borrow();
        let keys = self.keys.borrow();
        keys.get(*ix).and_then(|key| map.get(key).cloned())
    }
    pub(crate) fn remove(&self, key: &K) -> Option<RefCell<T>> {
        let obj = self.map.borrow_mut().remove(key);
        if obj.is_some() {
            self.keys.borrow_mut().retain(|k| k != key);
            obj
        } else {
            None
        }
    }
    pub(crate) fn contains(&self, key: &K) -> bool {
        self.map.borrow().contains_key(key)
    }
}

pub struct OrderedRefCellIterator<'a, K: Clone + Eq + std::hash::Hash + 'static, T: Clone> {
    list: &'a OrderedRefCell<K, T>,
    index: usize,
}

impl<K: Clone + Eq + std::hash::Hash, T: Clone> OrderedRefCell<K, T> {
    pub fn iter(&self) -> OrderedRefCellIterator<K, T> {
        OrderedRefCellIterator {
            list: self,
            index: 0,
        }
    }
}

impl<'a, K: Clone + Eq + std::hash::Hash, T: Clone> Iterator for OrderedRefCellIterator<'a, K, T> {
    type Item = (K, RefCell<T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.list.len() {
            return None;
        }
        let key = self.list.get_keys()[self.index].clone();
        self.index += 1;
        self.list.get(&key).map(|value| (key, value))
    }
}

// test
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Debug, Clone, PartialEq)]
    struct Neuron {
        pub key: String,
        pub value: Rc<RefCell<f32>>,
    }

    impl Default for Neuron {
        fn default() -> Self {
            Neuron {
                key: "".to_string(),
                value: Rc::new(RefCell::new(0.0)),
            }
        }
    }

    impl Neuron {
        pub fn new(output: bool, bias: f32) -> Self {
            let n = Neuron {
                ..Neuron::default()
            };
            n
        }
    }


    // #[test]
    fn test_neuron_manager_insert() {
        let n_manager = OrderedRefCell::new();
        assert_eq!(n_manager.insert("test".to_string(), RefCell::new(Neuron::new(false, 0.0))), None);
        assert_eq!(n_manager.insert("test".to_string(), RefCell::new(Neuron::new(false, 0.0))), Some(RefCell::new(Neuron::new(false, 0.0))));
    }

    #[test]
    fn test_neuron_manager_remove() {
        let n_manager = OrderedRefCell::new();
        n_manager.insert("test".to_string(), RefCell::new(Neuron::new(false, 0.0)));
        assert_eq!(n_manager.remove(&"test".to_string()).is_some(), true);
        assert_eq!(n_manager.remove(&"test".to_string()).is_none(), true);
    }
    #[test]
    fn test_neuron_manager_get() {
        let n_manager = OrderedRefCell::new();
        n_manager.insert("test".to_string(), RefCell::new(Neuron::new(false, 0.0)));
        assert_eq!(n_manager.get(&"test".to_string()).is_some(), true);
        assert_eq!(n_manager.get(&"nonexistent".to_string()).is_none(), true);
    }

    #[test]
    fn test_neuron_manager_contains() {
        let n_manager = OrderedRefCell::new();
        n_manager.insert("test".to_string(), RefCell::new(Neuron::new(false, 0.0)));
        assert_eq!(n_manager.contains(&"test".to_string()), true);
        assert_eq!(n_manager.contains(&"nonexistent".to_string()), false);
    }

    #[test]
    fn test_neuron_manager_len() {
        let n_manager = OrderedRefCell::new();
        assert_eq!(n_manager.len(), 0);
        n_manager.insert("test".to_string(), RefCell::new(Neuron::new(false, 0.0)));
        assert_eq!(n_manager.len(), 1);
    }

    #[test]
    fn test_neuron_manager_into_iterator() {
        let n_manager = OrderedRefCell::new();
        n_manager.insert("test1".to_string(), RefCell::new(Neuron::new(false, 0.0)));
        n_manager.insert("test2".to_string(), RefCell::new(Neuron::new(false, 0.0)));
        let mut iter = n_manager.iter();
        assert_eq!(iter.next().is_some(), true);
        assert_eq!(iter.next().is_some(), true);
        assert_eq!(iter.next().is_none(), true);
    }
    #[test]
    fn test_mutate_item_value() {
        let n_manager = OrderedRefCell::new();
        n_manager.insert("test1".to_string(), RefCell::new(Neuron::new(false, 0.0)));
        n_manager.insert("test2".to_string(), RefCell::new(Neuron::new(false, 0.0)));
        let mut iter = n_manager.iter();
        for (key, value) in iter {
            let mut neuron = value.borrow_mut();
            *neuron.value.borrow_mut() = 1.0;
            println!("{:?}", neuron.value.borrow());
        }
        let iter2 = n_manager.iter();
        for (key, value) in iter2 {
            let mut neuron = value.borrow();
            println!("{:?}", neuron.value.borrow());
            assert_eq!(*value.borrow().value.borrow(), 1.0);
        }
    }
    #[test]
    fn test_ordered_ref_cell_deep_clone() {
        let cell = OrderedRefCell::new();

        // Insert some elements
        cell.insert("key1".to_string(), RefCell::new(1));
        cell.insert("key2".to_string(), RefCell::new(2));

        // Deep clone the cell
        let cloned_cell = cell.deep_clone();

        // Check that the original and the clone have the same elements
        assert_eq!(cell.get(&"key1".to_string()), cloned_cell.get(&"key1".to_string()));
        assert_eq!(cell.get(&"key2".to_string()), cloned_cell.get(&"key2".to_string()));

        // Check that the original and the clone are not the same instance
        assert_ne!(&cell as *const _, &cloned_cell as *const _);
    }

}