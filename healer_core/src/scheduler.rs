use std::{
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{gen::choose_weighted, RngType};
// use healer_fuzzer::util;

#[derive(Debug, Default)]
pub struct Scheduler {
    // 0: static_relation, 1: model
    exec_total: [AtomicU64;2],
    intst_total: [AtomicU64;2],
    weights: [AtomicU64;2],
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            exec_total: [AtomicU64::new(0), AtomicU64::new(0)],
            intst_total: [AtomicU64::new(0), AtomicU64::new(0)],
            weights: [AtomicU64::new(1), AtomicU64::new(2)],
        }
        
    }
    pub fn inc_exec_total(&self, pos: usize) {
        self.exec_total[pos].fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_intst_total(&self, pos: usize) {
        self.intst_total[pos].fetch_add(1, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        for i in 0..2 {
            self.exec_total[i].store(0, Ordering::Relaxed);
            self.intst_total[i].store(0, Ordering::Relaxed);
            self.weights[i].store(i as u64+1,Ordering::Relaxed);
        }
    }

    pub fn update_with_ucb(&self) {
        let exec_total: f64 = self.exec_total.iter().map(|x| x.load(Ordering::Relaxed)).sum::<u64>() as f64;
        let mut max_val = 0;
        let mut max_index = 0;
        for i in 0..2 {
            let cur_total= self.exec_total[i].load(Ordering::Relaxed) as f64;
            let cur_intst= self.intst_total[i].load(Ordering::Relaxed) as f64;
            let item1 = (1000.0*cur_intst/cur_total) as u64 ;
            let item2 = (1000.0*((2.0 * (exec_total+1.0).ln() / (cur_total+1.0)).sqrt())) as u64;
            if item1+item2 > max_val {
                max_val = item1+item2;
                max_index = i;
            }
        }
        self.weights[0].store(1-max_index as u64, Ordering::Relaxed);
        self.weights[1].store(1, Ordering::Relaxed);
        // if max_index == 0 {
        //     self.weights[0].store(1, Ordering::Relaxed);
        //     self.weights[1].store(1, Ordering::Relaxed);
        // }
        // else {
        //     self.weights[0].store(0, Ordering::Relaxed);
        //     self.weights[1].store(1, Ordering::Relaxed);            
        // }
    }

    pub fn select(&self, rng: &mut RngType) -> usize {
        let weights = self.weights.iter().map(|w| w.load(Ordering::Relaxed)).collect::<Vec<u64>>();
        let idx = choose_weighted(rng, &weights);
        idx
    }
}
