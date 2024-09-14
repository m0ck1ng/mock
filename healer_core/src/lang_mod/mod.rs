#[cfg(test)]
mod tests {
    use tch::{Tensor, Device};

    use crate::lang_mod::{utils::accuracy};

    use super::model::ModelWrapper;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
    
    #[test]
    fn test_accuracy() {
        let b = Tensor::of_slice(&[0,1,2]);
        let a: Tensor = Tensor::stack(&vec![Tensor::of_slice(&[0.3,0.5,0.2]), Tensor::of_slice(&[0.8,0.2,0.0]), Tensor::of_slice(&[0.1,0.2,0.7])], 0);
        let topk = vec![1,2];
        let ret = accuracy(&a, &b, topk);
        assert_eq!(ret, vec![50.00, 100.00]);
    }

    #[test]
    fn test_train_model() {
        let model = ModelWrapper::default();
        let corpus = String::from("/home/workdir/output/corpus");
        let testcase = String::from("/home/workdir/output/corpus");
        let model_file = model.train(&corpus, &testcase);
        println!("model_file: {}", model_file);
        assert_eq!(model_file, String::from("/home/model_manager/api/lang_model/checkpoints/syscall_model_jit_best.pt"));
    }

    #[test]
    fn test_load_model() {
        let model = ModelWrapper::default();
        assert!(!model.exists());
        let model_file = "/home/model_manager/api/lang_model/checkpoints/syscall_model_jit_best.pt";
        model.load(model_file);
        assert!(model.exists());
    }

    #[test]
    fn test_eval_model() {
        let mut device = Device::Cpu;
        if tch::Cuda::is_available() {
            device = Device::Cuda(0);
        }
        println!("{:?}", device);
        assert!(device.is_cuda());
        let model = ModelWrapper::new("/home/model_manager/api/lang_model/checkpoints/syscall_model_jit_best.pt");
        let out = model.eval(&[0,1,2]).unwrap().to(device);
        assert_eq!(out.size(), vec![3,4130])
    }

    #[test]
    fn test_update_num() {
        let device = Device::cuda_if_available();
        println!("{:?}", device);
        assert!(!device.is_cuda());
        let model = ModelWrapper::new("/home/model_manager/api/lang_model/checkpoints/syscall_model_jit_best.pt");
        assert_eq!(model.update_num(), 0);
        let model_file = "/home/model_manager/api/lang_model/checkpoints/syscall_model_jit_best.pt";
        model.load(model_file);
        assert_eq!(model.update_num(), 1);
    }
    
    #[test]
    fn test_cuda() {
        let mut device = Device::Cpu;
        if tch::Cuda::is_available() {
            device = Device::Cuda(0);
        }
        println!("{:?}", device);
        assert!(device.is_cuda());
    }

}

mod utils;
pub mod model;
pub mod mutate;
