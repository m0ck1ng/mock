
use std::{collections::HashMap};

use rand::{prelude::SliceRandom};
use tch::Tensor;

use crate::{
    context::Context, 
    RngType, 
    syscall::SyscallId, 
    select::select_with_calls, mutation::seq::select_call_to,
};

pub fn select_call_to_wrapper(ctx: &mut Context, rng: &mut RngType, idx: usize) -> (SyscallId, usize) {
    type Selector = fn(&mut Context, &mut RngType, usize) -> SyscallId;
    const SELECTORS: [Selector; 2] = [select_call_to, select_call_to_lang];
    let mut pos = 0;
    if ctx.model.exists() {
        pos = ctx.scheduler.select(rng);
    } 
    (SELECTORS[pos](ctx, rng, idx), pos)
}

/// Select new call to location `idx`.
pub fn select_call_to_lang(ctx: &mut Context, rng: &mut RngType, idx: usize) -> SyscallId {
    let model = ctx.model().inner.read().unwrap();
    let calls = ctx.calls();

    // first, consider calls that can be influenced by calls before `idx`.
    // as in NLP, syscall are labeled(word_id) from "3", 0->3, 1->4,
    // so it require to cast between sid and word_id
    let topk = 10;
    let mut prev_calls: Vec<SyscallId> = calls[..idx].iter().map(|c| c.sid()+3).collect();
    // "2" refer to Start-Of-Sentence(SOS)
    prev_calls.insert(0, 1);
    let prev_pred = model.eval(&prev_calls).unwrap();
    let mut candidates = top_k(&prev_pred, topk);

    // then, consider calls that can be influence calls after `idx`.
    // cast between sid and word_id
    if idx != calls.len() {
        let mut back_calls: Vec<SyscallId> = calls[idx..].iter().rev().map(|c| c.sid()+3).collect();
        // "3" refer to End-of-Sentence(EOS)
        back_calls.insert(0, 2);
        let back_pred = model.eval(&back_calls).unwrap();
        candidates.extend(top_k(&back_pred, topk));
    }

    let candidates: Vec<(SyscallId, f64)> = candidates.into_iter().collect::<Vec<(SyscallId, f64)>>();
    if let Ok(candidate) = candidates.choose_weighted(rng, |candidate| candidate.1) {
        if candidate.0 >= 3 {
            candidate.0-3
        }
        else {
            // failed to select with relation, use normal strategy.
            select_with_calls(ctx, rng)
        }
    } else {
        // failed to select with relation, use normal strategy.
        select_with_calls(ctx, rng)
    }
}

// generate `topk` candidates for given distribution
pub fn top_k(pred: &Tensor, topk: i64) -> HashMap<usize, f64>{
    let (pred_val, pred_indexes) = pred.topk(topk, 1, true, true);
    let pred_val: Vec<f64> = Vec::from(pred_val);
    let pred_indexes: Vec<i16> = Vec::from(pred_indexes);
    let candidates: HashMap<SyscallId, f64> = pred_indexes.into_iter()
            .map(|i| i as SyscallId)
            .zip(pred_val.into_iter())
            .collect();
    candidates
}
