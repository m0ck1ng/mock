use criterion::{criterion_group, criterion_main, Criterion};
use healer_core::{
    gen::gen_prog,
    relation::{Relation, RelationWrapper}, lang_mod::model::ModelWrapper, scheduler::Scheduler,
};
use rand::{prelude::SmallRng, SeedableRng};
use syz_wrapper::sys::{load_sys_target, SysTarget};

pub fn bench_prog_gen(c: &mut Criterion) {
    let target = load_sys_target(SysTarget::LinuxAmd64).unwrap();
    let relation = Relation::new(&target);
    let rw = RelationWrapper::new(relation);
    let mw = ModelWrapper::default();
    let sc = Scheduler::default();
    let mut rng = SmallRng::from_entropy();

    c.bench_function("prog-gen", |b| b.iter(|| gen_prog(&target, &rw, &mw, &sc, &mut rng)));
}

criterion_group!(benches, bench_prog_gen);
criterion_main!(benches);
