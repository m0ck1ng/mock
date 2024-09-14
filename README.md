# MOCK

We open source the prototype of Mock here. Mock is a Linux kernel fuzzer that can learn the contextual dependencies in syscall sequences and then generate context-aware test cases. In this way, Mock improves the input quality and explore deeper space of kernels. More details can be found in [our paper](https://www.ndss-symposium.org/ndss-paper/mock-optimizing-kernel-fuzzing-mutation-with-context-aware-dependency/) from NDSS'24.

# Installation

As Mock is built upon [Healer](https://github.com/SunHao-0/healer), please follow the [instructions](https://github.com/SunHao-0/healer/blob/main/README.md) to prepare necessary toolchains. Image and kernel preparation can be found in this [document](https://github.com/google/syzkaller/blob/master/docs/linux/setup_ubuntu-host_qemu-vm_x86-64-kernel.md).

Besides, the training component is written in Python and interacts with the fuzzing component via http. Therefore, Python packages should be installed.
```
> pip3 install numpy django 
# if cuda is available
> pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu113
```

Once all the required tools have been installed, Mock can be easily built using the following command. It may some take to prepare the Rust bindings for Libtorch, [tch-rc](https://github.com/LaurentMazare/tch-rs), on which Mock depends.
```
> cargo build --release
```
You can find Mock and the patched Syzkaller binary (`syz-bin`) can be found in the `target/release` directory.

# Usage

The first step is run Mock is to launch a django service. It serves to monitor requests for the parallel model training. Start it using the following command:

```
> cd $MOCK_ROOT/tools/model_manager
> python3 manage.py runserver
```

Suppose that everything is fine ([instructions](https://github.com/m0ck1ng/mock_private/blob/main/README.md#fuzz-linux-kernel-with-healer) here), execute following command in a new terminal to start the fuzzing (`-d` specifies the path to disk image, `-k` specifies the path to kernel image and `--ssh-key` specifies the path to ssh key).

```
> healer -d stretch.img --ssh-key stretch.id_rsa -k bzImage
```

# Citation
```
@inproceedings{
    author = {Jiacheng, Xu and Xuhong, Zhang and Shouling, Ji and Yuan, Tian and Binbin, Zhao and Qinying, Wang and Peng, Cheng and Jiming, Chen}, 
    title = {MOCK: Optimizing Kernel Fuzzing Mutation with Context-aware Dependency},
    booktitle = {31st Annual Network and Distributed System Security Symposium, {NDSS} 2024, San Diego, California, USA, February 26 - March 1, 2024}, 
    publisher = {The Internet Society},
    year = {2024}, 
}
```