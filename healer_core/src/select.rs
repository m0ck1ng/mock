//! Call Selection.
//!
//! Select syscalls based on syscalls, syscall => input/output resources, resources => input/output syscalls,
//! resource => sub/super types, relations.
use crate::{context::Context, gen::choose_weighted, syscall::SyscallId, HashMap, RngType, lang_mod::mutate::top_k};
use rand::prelude::*;

/// Select a syscall based on current calls context.
#[inline]
pub fn select(ctx: &Context, rng: &mut RngType) -> SyscallId {
    if ctx.calls().is_empty() {
        select_with_no_calls(ctx, rng)
    } else {
        select_with_calls(ctx, rng)
    }
}

/// Weight for selector.
type Weight = u64;
type Selector = fn(&Context, &mut RngType) -> Option<SyscallId>;

/// Select a syscall without any initial call
pub fn select_with_no_calls(ctx: &Context, rng: &mut RngType) -> SyscallId {
    /// Selectors that can be used when initial calls are given.
    const SELECTORS: [Selector; 3] = [
        select_res_output_syscall, // 80%
        select_calls_wrapper,      // 10%
        select_random_syscall,     // 10%
    ];
    const WEIGHTS: [Weight; 3] = [80, 90, 100];

    loop {
        let idx = choose_weighted(rng, &WEIGHTS);
        if let Some(sid) = SELECTORS[idx](ctx, rng) {
            debug_info!("select with no calls, strategy-{}: {}", idx, ctx.target().syscall_of(sid));
            return sid;
        }
    }
    
    // loop {
    //     let selector = if rng.gen_ratio(9, 10) {
    //         select_res_output_syscall
    //     } else {
    //         select_random_syscall
    //     };
    //     if let Some(sid) = selector(ctx, rng) {
    //         debug_info!("select with no calls: {}", ctx.target().syscall_of(sid));
    //         return sid;
    //     }
    // }
}

/// Select based on generated calls.
pub fn select_with_calls(ctx: &Context, rng: &mut RngType) -> SyscallId {
    /// Selectors that can be used when initial calls are given.
    const SELECTORS: [Selector; 4] = [
        select_calls_wrapper,      // 50%
        select_res_input_syscall,  // 20%
        select_res_output_syscall, // 15%
        select_random_syscall,     // 10%
    ];
    const WEIGHTS: [Weight; 4] = [60, 80, 95, 100];

    loop {
        let idx = choose_weighted(rng, &WEIGHTS);
        if let Some(sid) = SELECTORS[idx](ctx, rng) {
            debug_info!("select strategy-{}: {}", idx, ctx.target().syscall_of(sid));
            return sid;
        }
    }
}

#[inline]
pub fn select_calls_wrapper(ctx: &Context, rng: &mut RngType) -> Option<usize> {
    if ctx.model().exists() {
        select_with_model(ctx, rng)
    } 
    else {
        select_with_relation(ctx, rng)
    }
}

/// Select based on generated calls and relations.
pub fn select_with_relation(ctx: &Context, rng: &mut RngType) -> Option<SyscallId> {
    let mut candidates: HashMap<SyscallId, Weight> = HashMap::new();
    let r = ctx.relation().inner.read().unwrap();
    let calls = ctx.calls();

    for sid in calls.iter().map(|c| c.sid()) {
        for candidate in r.influence_of(sid).iter().copied() {
            let entry = candidates.entry(candidate).or_default();
            *entry += 1;
        }
    }

    let candidates: Vec<(SyscallId, Weight)> = candidates.into_iter().collect();
    candidates
        .choose_weighted(rng, |candidate| candidate.1)
        .ok()
        .map(|candidate| candidate.0)
}

/// Select based on generated calls and model.
pub fn select_with_model(ctx: &Context, rng: &mut RngType) -> Option<SyscallId> {
    let model = ctx.model().inner.read().unwrap();
    let calls = ctx.calls();

    let topk = 10;
    let mut prev_calls: Vec<SyscallId> = calls.iter().map(|c| c.sid()+3).collect();
    // "2" refer to Start-Of-Sentence(SOS)
    prev_calls.insert(0, 2);
    let prev_pred = model.eval(&prev_calls).unwrap();
    let candidates = top_k(&prev_pred, topk);
    let candidates: Vec<(SyscallId, f64)> = candidates.into_iter().collect::<Vec<(SyscallId, f64)>>();
    if let Some(sid) = candidates
        .choose_weighted(rng, |candidate| candidate.1)
        .ok()
        .map(|candidate| candidate.0) {
        if sid >= 3 {
            return Some(sid-3)
        }
        else {
            return None
        }
    }
    None
}

/// Select syscall that can output resources.
pub fn select_res_output_syscall(ctx: &Context, rng: &mut RngType) -> Option<SyscallId> {
    let selected_res_kind = if ctx.res().is_empty() || rng.gen_ratio(1, 5) {
        ctx.target().res_kinds().choose(rng).unwrap()
    } else {
        let res_base = ctx.res().choose(rng).unwrap();
        if rng.gen() {
            ctx.target().res_sub_tys(res_base).choose(rng).unwrap()
        } else {
            ctx.target().res_super_tys(res_base).choose(rng).unwrap()
        }
    };
    ctx.target()
        .res_output_syscall(selected_res_kind)
        .choose(rng)
        .copied()
}

/// Select syscall that consume current resource
pub fn select_res_input_syscall(ctx: &Context, rng: &mut RngType) -> Option<SyscallId> {
    if let Some(mut res) = ctx.res().choose(rng) {
        if rng.gen_ratio(3, 10) {
            // use syscalls that take super type of 'res' as input
            res = ctx.target().res_sub_tys(res).choose(rng).unwrap();
        }
        ctx.target().res_input_syscall(res).choose(rng).copied()
    } else {
        None
    }
}

/// Select syscall randomly
pub fn select_random_syscall(ctx: &Context, rng: &mut RngType) -> Option<SyscallId> {
    ctx.target().enabled_syscalls().choose(rng).map(|s| s.id())
}
