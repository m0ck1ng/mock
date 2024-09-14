use std::{sync::RwLock, collections::HashMap, time::Duration};
use reqwest;
use tch::{CModule, Tensor, Device};

#[derive(Default, Debug)]
pub struct ModelWrapper {
    pub inner: RwLock<Model>,
}

impl ModelWrapper {
    pub fn new <P: AsRef<std::path::Path>>(path: P) -> Self {
        Self {
            inner:  RwLock::new(Model::new(path))
        }
    }
    
    #[inline]
    pub fn load<P: AsRef<std::path::Path>>(&self, path: P) {
        let mut inner = self.inner.write().unwrap();
        inner.load(path);
    }

    pub fn train(&self, corpus: &String, testcase: &String) -> String {
        let mut params = HashMap::new();
        params.insert("corpus", corpus.clone());
        params.insert("testcase", testcase.clone());

        let client = reqwest::blocking::Client::new();
        let res = client.post("http://127.0.0.1:8000/api/model")
            .timeout(Duration::from_secs(60*60))
            .form(&params).send().unwrap();
        let model_file = res.text().unwrap();
        model_file
    }

    #[inline]
    pub fn eval(&self, inputs: &[usize]) -> Option<Tensor> {
        let inner = self.inner.read().unwrap();
        inner.eval(inputs)
    }

    #[inline]
    pub fn exists(&self) -> bool {
        let inner = self.inner.read().unwrap();
        !inner.model.is_none()
    }

    #[inline]
    pub fn update_num(&self) -> u32 {
        let inner = self.inner.read().unwrap();
        inner.update_num()
    }
}

#[derive(Default, Debug)]
pub struct Model {
    model: Option<CModule>,
    device: Option<Device>,
    update_num: u32,
}

impl Model {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Self {
        let mut device = Device::Cpu;
        if tch::Cuda::is_available() {
            device = Device::Cuda(0);
        }
        Self {
            model: Some(tch::CModule::load_on_device(path, device).unwrap()),
            device: Some(device),
            update_num: 0,
        }
    }
    
    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) {
        let mut device = Device::Cpu;
        if tch::Cuda::is_available() {
            device = Device::Cuda(0);
        }
        self.device = Some(device);
        self.model = Some(tch::CModule::load_on_device(path, device).unwrap());
        self.update_num = self.update_num + 1;
    }

    pub fn eval(&self, inputs: &[usize]) -> Option<Tensor> {
        if self.model.is_none() {
            None
        }
        else {
            let length = Tensor::of_slice(&[inputs.len() as u8]).to_device(self.device.unwrap());
            let inputs_i32 = inputs.into_iter().map(|i| *i as i32).collect::<Vec<i32>>();
            let inputs_i32 = Tensor::stack(&[Tensor::of_slice(&inputs_i32)], 0).to_device(self.device.unwrap());
            let outputs = self.model.as_ref().unwrap().forward_ts(&[inputs_i32, length]).unwrap().softmax(-1, tch::Kind::Float);
            Some(outputs)
        }
    }

    #[inline]
    pub fn update_num(&self) -> u32 {
        self.update_num

    }
}
