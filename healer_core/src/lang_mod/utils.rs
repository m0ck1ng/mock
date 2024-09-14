use std::{
    fs::{read_to_string}
};
use tch::{Tensor, IndexOp, kind};

#[allow(dead_code)]
pub fn read_syscall<P: AsRef<std::path::Path>>(path: P) -> Vec<String>{
    let content = read_to_string(path).unwrap();
    let syscalls = content.split(" ").map(|x| x.to_string()).collect();
    syscalls
}

#[allow(dead_code)]
pub fn accuracy(predict: &Tensor, labels: &Tensor, topk: Vec<i64>) -> Vec<f64>{
    let maxk = *topk.last().unwrap();
    let pred = predict.topk(maxk, 1, true, true);
    let pred_indexes = pred.1.transpose(1, 0);
    
    let pad_mask = Tensor::zeros(&labels.size(), kind::INT64_CPU);
    let pad_mask = pad_mask.ne_tensor(&labels);

    let correct = pred_indexes.eq_tensor(&labels.view([1,-1]).expand_as(&pred_indexes));
    let correct = correct.logical_and(&pad_mask);

    let total = f64::from(pad_mask.sum(tch::Kind::Float));
    let mut ret: Vec<f64> = Vec::new();
    for k in topk {
        let correct_k = correct.i(0..k).reshape(&[1,-1]).sum(tch::Kind::Float);
        let val = f64::trunc(f64::from(correct_k*100.0*100.0/total))/100.0;
        ret.push(val);
    }
    ret
}

#[allow(dead_code)]
pub struct  AverageMetrics {
    total: f64,
    cnt: usize
}

#[allow(dead_code)]
impl AverageMetrics {
    pub fn new() -> Self {
        AverageMetrics { total: 0., cnt: 0}
    }

    pub fn update(&mut self, loss: f64, cnt: usize) {
        self.total += loss;
        self.cnt += cnt;
    }

    pub fn avg(&self) -> f64 {
        self.total / self.cnt as f64
    }

    pub fn reset(&mut self) {
        self.total = 0.;
        self.cnt = 0;
    }
}
