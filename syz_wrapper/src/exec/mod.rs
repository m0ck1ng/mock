//! Syz-executor driver.
#![allow(clippy::or_fun_call, clippy::redundant_slicing)]

pub mod features;
pub mod message;
pub mod serialization;
mod util;

use crate::{
    exec::{message::*, serialization::*, util::*},
    HashMap, HashSet,
};
use healer_core::{prog::Prog, target::Target};
use healer_io::{thread::read_background, BackgroundIoHandle};
use iota::iota;
use shared_memory::Shmem;
use std::io::Read;
use std::os::unix::net::UnixStream;
use std::{io::ErrorKind, time::Duration};
use std::{
    io::Write,
    mem::size_of_val,
    process::{Child, Command, Stdio},
};
use thiserror::Error;
use timeout_readwrite::TimeoutReader;

use self::features::Features;

/// Env flags to executor.
pub type EnvFlags = u64;

iota! {
    const FLAG_DEBUG: EnvFlags = 1 << (iota);             // debug output from executor
    , FLAG_SIGNAL                                    // collect feedback signals (coverage)
    , FLAG_SANDBOX_SETUID                            // impersonate nobody user
    , FLAG_SANDBOX_NAMESPACE                         // use namespaces for sandboxing
    , FLAG_SANDBOX_ANDROID                           // use Android sandboxing for the untrusted_app domain
    , FLAG_EXTRA_COVER                               // collect extra coverage
    , FLAG_ENABLE_TUN                                // setup and use /dev/tun for packet injection
    , FLAG_ENABLE_NETDEV                             // setup more network devices for testing
    , FLAG_ENABLE_NETRESET                           // reset network namespace between programs
    , FLAG_ENABLE_CGROUPS                            // setup cgroups for testing
    , FLAG_ENABLE_CLOSEFDS                           // close fds after each program
    , FLAG_ENABLE_DEVLINKPCI                         // setup devlink PCI device
    , FLAG_ENABLE_VHCI_INJECTION                     // setup and use /dev/vhci for hci packet injection
    , FLAG_ENABLE_WIFI                               // setup and use mac80211_hwsim for wifi emulation
}

pub fn default_env_flags(debug: bool, sandbox: &str) -> EnvFlags {
    let mut env = FLAG_SIGNAL;
    env |= sandbox_to_flags(sandbox);
    if debug {
        env |= FLAG_DEBUG;
    }
    env
}

pub fn sandbox_to_flags(sandbox: &str) -> EnvFlags {
    match sandbox {
        "setuid" => FLAG_SANDBOX_SETUID,
        "namespace" => FLAG_SANDBOX_NAMESPACE,
        "android" => FLAG_SANDBOX_ANDROID,
        _ => 0,
    }
}

pub fn flags_to_sandbox(env: EnvFlags) -> String {
    if env & FLAG_SANDBOX_SETUID != 0 {
        "setuid".to_string()
    } else if env & FLAG_SANDBOX_NAMESPACE != 0 {
        "namespace".to_string()
    } else if env & FLAG_SANDBOX_ANDROID != 0 {
        "android".to_string()
    } else {
        "none".to_string()
    }
}

/// Flag for controlling execution behavior.
pub type ExecFlags = u64;

iota! {
    pub const FLAG_COLLECT_COVER : ExecFlags = 1 << (iota);       // collect coverage
    , FLAG_DEDUP_COVER                                 // deduplicate coverage in executor
    , FLAG_INJECT_FAULT                                // inject a fault in this execution (see ExecOpts)
    , FLAG_COLLECT_COMPS                               // collect KCOV comparisons
    , FLAG_THREADED                                    // use multiple threads to mitigate blocked syscalls
    , FLAG_COLLIDE                                     // collide syscalls to provoke data races
    , FLAG_ENABLE_COVERAGE_FILTER                      // setup and use bitmap to do coverage filter
}

/// Option for controlling execution behavior.
#[derive(Debug, Clone)]
pub struct ExecOpt {
    /// Options for this execution.
    pub flags: ExecFlags,
    /// Inject fault for 'fault_call'.
    pub fault_call: i32,
    /// Inject fault 'nth' for 'fault_call'
    pub fault_nth: i32,
}

impl Default for ExecOpt {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecOpt {
    pub const fn new() -> Self {
        Self {
            flags: FLAG_DEDUP_COVER | FLAG_THREADED | FLAG_COLLIDE,
            fault_call: 0,
            fault_nth: 0,
        }
    }

    #[inline]
    pub fn disable(&mut self, flag: u64) {
        self.flags &= u64::MAX ^ flag;
    }

    #[inline]
    pub fn enable(&mut self, flag: u64) {
        self.flags |= flag;
    }
}

/// Flag for execution result of one call.
pub type CallFlags = u32;

iota! {
    pub const CALL_EXECUTED : CallFlags = 1 << (iota); // started at all
    , CALL_FINISHED                                // finished executing (rather than blocked forever)
    , CALL_BLOCKED                                 // finished but blocked during execution
    , CALL_FAULT_INJECTED                          // fault was injected into this call
}

/// Execution of one call.
#[derive(Debug, Default, Clone)]
pub struct CallExecInfo<'a, 'b> {
    pub flags: CallFlags,
    /// Branch coverage.
    pub branches: &'a [u32],
    /// Block converage.
    pub blocks: &'b [u32],
    /// per-call comparison operands
    pub comp_map: HashMap<u64, HashSet<u64>>,
    /// Syscall errno, indicating the success or failure.
    pub errno: i32,
}

#[derive(Debug, Default, Clone)]
pub struct ExtraCallExecInfo {
    /// Branch coverage.
    pub branches: HashSet<u32>,
    /// Block converage.
    pub blocks: HashSet<u32>,
}

#[derive(Debug, Default, Clone)]
pub struct ProgExecInfo<'a, 'b> {
    pub call_infos: Vec<CallExecInfo<'a, 'b>>,
    pub extra: Option<ExtraCallExecInfo>,
}

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("prog serialization: {0}")]
    ProgSerialization(#[from] SerializeError),
    #[error("exec internal")]
    ExecInternal,
    #[error("killed(maybe cause by timeout)")]
    TimeOut,
    #[error("unexpected executor exit status: {0}")]
    UnexpectedExitStatus(i32),
    #[error("output parse: {0}")]
    OutputParse(String),
}

#[derive(Debug, Error)]
pub enum SpawnError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("spawn: {0}")]
    Spawn(String),
    #[error("handshake: {0}")]
    HandShake(std::io::Error),
}

/// Size of syz-executor input shared memory.
pub const IN_SHM_SZ: usize = 4 << 20;
/// Size of syz-executor output shared memory.
pub const OUT_SHM_SZ: usize = 16 << 20;

/// Config for executor, auto detected.
pub struct ExecConfig {
    pub pid: u64,
    pub env: EnvFlags,
    pub features: Features,
    pub shms: Option<(Shmem, Shmem)>,
    pub unix_socks: Option<(String, String, String)>,
    // stdin, stdout, stderr
    pub use_forksrv: bool,
    pub debug: bool,
}

impl Clone for ExecConfig {
    fn clone(&self) -> Self {
        Self {
            pid: self.pid,
            env: self.env,
            features: self.features,
            shms: None,
            use_forksrv: self.use_forksrv,
            unix_socks: self.unix_socks.clone(),
            debug: self.debug,
        }
    }
}

pub struct ExecutorHandle {
    pid: u64,
    env: EnvFlags,

    use_shm: bool,
    use_forksrv: bool,
    use_extern_chan: bool,
    in_shm: Option<Shmem>,
    out_shm: Option<Shmem>,
    in_mem: Option<Box<[u8]>>,
    out_mem: Option<Box<[u8]>>,

    cmd: Option<Command>,
    exec_child: Option<Child>,
    exec_stdin: Option<Box<dyn Write + 'static>>,
    exec_stdout: Option<Box<dyn Read + 'static>>,
    exec_stderr: Option<BackgroundIoHandle>,

    debug: bool,
}

unsafe impl Send for ExecutorHandle {}

unsafe impl Sync for ExecutorHandle {}

impl ExecutorHandle {
    pub fn with_config(config: ExecConfig) -> Self {
        let use_shm = config.shms.is_some();
        let (mut in_shm, mut out_shm) = (None, None);
        let (mut in_mem, mut out_mem) = (None, None);
        if use_shm {
            let shms = config.shms.unwrap();
            in_shm = Some(shms.0);
            out_shm = Some(shms.1);
        } else {
            in_mem = Some(vec![0; IN_SHM_SZ].into_boxed_slice());
            out_mem = Some(vec![0; OUT_SHM_SZ].into_boxed_slice());
        };

        Self {
            pid: config.pid,
            use_shm,
            use_forksrv: config.use_forksrv,
            use_extern_chan: config.unix_socks.is_some(),
            in_shm,
            out_shm,
            in_mem,
            out_mem,
            env: config.env,
            cmd: None,
            exec_child: None,
            exec_stdin: None,
            exec_stdout: None,
            exec_stderr: None,
            debug: config.debug,
        }
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    pub fn execute_one<'a>(
        &'a mut self,
        target: &Target,
        p: &Prog,
        opt: &ExecOpt,
    ) -> Result<ProgExecInfo<'a, 'a>, ExecError> {
        if let Err(ExecError::Io(e)) = self.req_exec(target, p, opt) {
            log::debug!("failed to send exec req: {}", e);
            self.kill();
            return Err(ExecError::Io(e));
        }

        if let Err(e) = self.wait_finish() {
            log::debug!("error happened during waiting exec finish: {}", e);
            self.kill();
            return Err(e);
        }

        self.parse_output(p)
    }

    pub fn respawn(&mut self) -> Result<(), SpawnError> {
        if self.use_extern_chan {
            self.do_bg_spawn()?;
            if self.use_forksrv {
                self.handshake()?;
            }
            Ok(())
        } else {
            let cmd = self.cmd.as_ref().unwrap();
            let mut exec_cmd = Command::new(cmd.get_program());
            exec_cmd.args(cmd.get_args());
            for (k, v) in cmd.get_envs().filter(|(_, v)| v.is_some()) {
                exec_cmd.env(k, v.unwrap());
            }
            if let Some(dir) = cmd.get_current_dir() {
                exec_cmd.current_dir(dir);
            }
            self.spawn(exec_cmd)
        }
    }

    pub fn bg_spawn(&mut self, exec_cmd: Command) -> Result<(), SpawnError> {
        self.cmd = Some(exec_cmd);
        self.do_bg_spawn()?;

        if self.use_forksrv {
            self.handshake()?;
        }

        Ok(())
    }

    pub fn spawn_with_channel(
        &mut self,
        exec_cmd: Command,
        (stdin, stdout, stderr): (UnixStream, UnixStream, Option<UnixStream>),
    ) -> Result<(), SpawnError> {
        assert!(self.use_extern_chan);
        self.cmd = None;
        self.reset();

        stdout
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        self.exec_stdin = Some(Box::new(stdin));
        self.exec_stdout = Some(Box::new(stdout));
        if let Some(mut stderr) = stderr {
            if self.debug {
                self.exec_stderr = Some(read_background(stderr, self.debug));
            } else {
                self.exec_stderr = None;
                std::thread::spawn(move || {
                    let mut sink = std::io::sink();
                    let _ = std::io::copy(&mut stderr, &mut sink);
                });
            }
        }

        self.bg_spawn(exec_cmd)
    }

    fn do_bg_spawn(&mut self) -> Result<(), SpawnError> {
        let exec_cmd = self.cmd.as_mut().unwrap();

        exec_cmd.stdin(Stdio::null());
        exec_cmd.stdout(Stdio::piped());
        exec_cmd.stderr(Stdio::piped());

        let output = exec_cmd.output()?;
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr).into_owned();
            Err(SpawnError::Spawn(err))
        } else {
            if self.debug {
                let out = String::from_utf8_lossy(&output.stdout);
                println!("{:?}: {}", exec_cmd, out);
            }
            Ok(())
        }
    }

    pub fn spawn(&mut self, mut exec_cmd: Command) -> Result<(), SpawnError> {
        assert!(!self.use_extern_chan);
        self.kill();

        exec_cmd.stdin(Stdio::piped());
        exec_cmd.stdout(Stdio::piped());
        if self.debug {
            exec_cmd.stderr(Stdio::piped());
        } else {
            exec_cmd.stderr(Stdio::null());
        }

        let mut child = exec_cmd.spawn()?;
        self.exec_stdin = Some(Box::new(child.stdin.take().unwrap()));
        let stdout = TimeoutReader::new(child.stdout.take().unwrap(), Duration::from_secs(30));
        self.exec_stdout = Some(Box::new(stdout));
        if let Some(stderr) = child.stderr.take() {
            self.exec_stderr = Some(read_background(stderr, self.debug));
        }
        self.exec_child = Some(child);

        if self.use_forksrv {
            if let Err(e) = self.handshake() {
                self.kill();
                return Err(e);
            }
        }

        self.cmd = Some(exec_cmd);
        Ok(())
    }

    fn handshake(&mut self) -> Result<(), SpawnError> {
        let req = HandshakeReq {
            magic: IN_MAGIC,
            env_flags: self.env,
            pid: self.pid,
        };
        log::debug!("{:?}", req);
        write_all(&mut self.exec_stdin.as_mut().unwrap(), &req).map_err(SpawnError::HandShake)?;

        log::debug!("exec-{}: waiting...", self.pid);
        let reply: HandshakeReply =
            read_exact(&mut self.exec_stdout.as_mut().unwrap()).map_err(SpawnError::HandShake)?;
        if reply.magic != OUT_MAGIC {
            panic!(
                "handshake reply magic not match, require: {:x}, got: {:x}",
                OUT_MAGIC, reply.magic
            );
        }

        Ok(())
    }

    #[inline]
    fn req_exec(&mut self, target: &Target, p: &Prog, opt: &ExecOpt) -> Result<(), ExecError> {
        let in_buf = self
            .in_shm
            .as_mut()
            .map(|shm| unsafe { shm.as_slice_mut() })
            .or(self.in_mem.as_deref_mut())
            .unwrap();
        let prog_sz = match serialize(target, p, in_buf) {
            Ok(left_sz) => in_buf.len() - left_sz,
            Err(e) => return Err(ExecError::ProgSerialization(e)),
        };

        let exec_req = ExecuteReq {
            env_flags: self.env,
            magic: IN_MAGIC,
            exec_flags: opt.flags,
            pid: self.pid,
            fault_call: opt.fault_call as u64,
            fault_nth: opt.fault_nth as u64,
            syscall_timeout_ms: 100,
            program_timeout_ms: 5000,
            slowdown_scale: 1,
            prog_size: if self.use_shm { 0 } else { prog_sz as u64 },
        };

        write_all(self.exec_stdin.as_mut().unwrap(), &exec_req)?;
        if !self.use_shm {
            self.exec_stdin
                .as_mut()
                .unwrap()
                .write_all(&in_buf[..prog_sz])?;
        }

        Ok(())
    }

    fn wait_finish(&mut self) -> Result<(), ExecError> {
        const SYZ_STATUS_INTERNAL_ERROR: i32 = 67;

        let mut out_buf = self
            .out_shm
            .as_mut()
            .map(|shm| unsafe { shm.as_slice_mut() })
            .or(self.out_mem.as_deref_mut())
            .unwrap();
        out_buf[0..4].iter_mut().for_each(|v| *v = 0);
        out_buf = &mut out_buf[4..];

        let exit_status;
        let mut exec_reply: ExecuteReply;
        loop {
            exec_reply = match read_exact(self.exec_stdout.as_mut().unwrap()) {
                Ok(r) => r,
                Err(e) => return Err(self.handle_possible_timeout(e)),
            };
            if exec_reply.magic != OUT_MAGIC {
                panic!(
                    "reply magic not match, required: {}, got: {}",
                    OUT_MAGIC, exec_reply.magic
                )
            }

            if exec_reply.done != 0 {
                exit_status = exec_reply.status as i32;
                break;
            }

            let r: CallReply = match read_exact(self.exec_stdout.as_mut().unwrap()) {
                Ok(r) => r,
                Err(e) => return Err(self.handle_possible_timeout(e)),
            };
            write_all(&mut out_buf[..], &r).unwrap();
            out_buf = &mut out_buf[size_of_val(&r)..];
        }

        match exit_status {
            0 => Ok(()),
            SYZ_STATUS_INTERNAL_ERROR => Err(ExecError::ExecInternal),
            _ => Err(ExecError::UnexpectedExitStatus(exit_status)),
        }
    }

    fn parse_output(&self, p: &Prog) -> Result<ProgExecInfo, ExecError> {
        const EXTRA_REPLY_INDEX: u32 = 0xffffffff;

        let mut out_buf = self
            .out_shm
            .as_ref()
            .map(|shm| unsafe { shm.as_slice() })
            .or(self.out_mem.as_deref())
            .unwrap();
        let ncmd = read_u32(&mut out_buf)
            .ok_or_else(|| ExecError::OutputParse("failed to read number of calls".to_string()))?;
        let mut call_infos = vec![CallExecInfo::default(); p.calls().len()];
        let mut extra = Vec::new();

        for i in 0..ncmd {
            let reply: &CallReply = read(&mut out_buf).ok_or_else(|| {
                ExecError::OutputParse(format!("failed to read call {} reply", i))
            })?;
            let call_info;
            if reply.index != EXTRA_REPLY_INDEX {
                if reply.index as usize >= call_infos.len() {
                    return Err(ExecError::OutputParse(format!(
                        "bad call {} index {}/{}",
                        i,
                        reply.index,
                        call_infos.len()
                    )));
                }
                let sid = p.calls()[reply.index as usize].sid();
                if sid != reply.num as usize {
                    return Err(ExecError::OutputParse(format!(
                        "wrong call {} num {}/{}",
                        i, reply.num, sid
                    )));
                }
                call_info = &mut call_infos[reply.index as usize];
                if call_info.flags != 0 || !call_info.branches.is_empty() {
                    return Err(ExecError::OutputParse(format!(
                        "duplicate reply for call {}/{}/{}",
                        i, reply.index, reply.num
                    )));
                }

                if reply.comps_size != 0 {
                    return Err(ExecError::OutputParse(format!(
                        "comparison collected for call {}/{}/{}",
                        i, reply.index, reply.num
                    )));
                }

                call_info.flags = reply.flags;
                call_info.errno = reply.errno as i32;
            } else {
                extra.push(CallExecInfo::default());
                call_info = extra.last_mut().unwrap();
            }

            if reply.branch_size != 0 {
                call_info.branches = read_u32_slice(&mut out_buf, reply.branch_size as usize)
                    .ok_or_else(|| {
                        ExecError::OutputParse(format!(
                            "call {}/{}/{}: signal overflow: {}/{}",
                            i,
                            reply.index,
                            reply.num,
                            reply.branch_size,
                            out_buf.len()
                        ))
                    })?;
            }
            if reply.block_size != 0 {
                call_info.blocks = read_u32_slice(&mut out_buf, reply.block_size as usize)
                    .ok_or_else(|| {
                        ExecError::OutputParse(format!(
                            "call {}/{}/{}: cover overflow: {}/{}",
                            i,
                            reply.index,
                            reply.num,
                            reply.block_size,
                            out_buf.len()
                        ))
                    })?;
            }
        }
        Ok(ProgExecInfo {
            call_infos,
            extra: if extra.is_empty() {
                None
            } else {
                Some(self.merge_extra(extra))
            },
        })
    }

    fn handle_possible_timeout(&mut self, e: std::io::Error) -> ExecError {
        if e.kind() == ErrorKind::TimedOut {
            log::debug!("executor timeout");
            ExecError::TimeOut
        } else {
            ExecError::Io(e)
        }
    }

    #[inline]
    fn kill(&mut self) {
        if let Some(child) = self.exec_child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
            self.exec_child = None;
        }
        if !self.use_extern_chan {
            self.reset();
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.exec_stdin = None;
        self.exec_stdout = None;
        self.exec_stderr = None;
    }

    fn merge_extra(&self, extras: Vec<CallExecInfo>) -> ExtraCallExecInfo {
        let mut br = HashSet::new();
        let mut bb = HashSet::new();
        for e in extras {
            br.extend(e.branches);
            bb.extend(e.branches);
        }
        ExtraCallExecInfo {
            branches: br,
            blocks: bb,
        }
    }
}

impl Drop for ExecutorHandle {
    fn drop(&mut self) {
        self.kill();
    }
}
