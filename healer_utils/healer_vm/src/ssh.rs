use std::path::Path;
use std::process::Command;

use thiserror::Error;

pub fn ssh_basic_cmd<T: AsRef<str>>(
    ip: T,
    port: u16,
    key: T, // use key
    user: T,
) -> Command {
    let mut ssh_cmd = Command::new("ssh");
    ssh_cmd
        .args(vec!["-F", "/dev/null"])
        .args(vec!["-o", "BatchMode=yes"])
        .args(vec!["-o", "IdentitiesOnly=yes"])
        .args(vec!["-o", "StrictHostKeyChecking=no"])
        .args(vec!["-o", "UserKnownHostsFile=/dev/null"])
        .args(vec!["-o", "ConnectTimeout=3s"])
        .args(vec!["-i", key.as_ref()])
        .args(vec!["-p", &format!("{}", port)])
        .arg(format!("{}@{}", user.as_ref(), ip.as_ref()));
    ssh_cmd
}

#[derive(Debug, Error)]
pub enum ScpError {
    #[error("scp: {0}")]
    Scp(String),
    #[error("spawn: {0}")]
    Spawn(#[from] std::io::Error),
}

pub fn scp<T: AsRef<str>, P: AsRef<Path>>(
    ip: T,
    port: u16,
    key: T,
    user: T,
    from: P,
    to: P,
) -> Result<(), ScpError> {
    let mut scp_cmd = Command::new("scp");
    let from = from.as_ref().display().to_string();
    let to = to.as_ref().display().to_string();

    scp_cmd
        .args(vec!["-F", "/dev/null"])
        .args(vec!["-o", "BatchMode=yes"])
        .args(vec!["-o", "IdentitiesOnly=yes"])
        .args(vec!["-o", "StrictHostKeyChecking=no"])
        .args(vec!["-o", "UserKnownHostsFile=/dev/null"])
        .args(vec!["-o", "ConnectTimeout=10s"])
        .args(vec!["-i", key.as_ref()])
        .args(vec!["-P", &format!("{}", port)])
        .arg(&from)
        .arg(format!("{}@{}:{}", user.as_ref(), ip.as_ref(), to));

    let output = scp_cmd.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8(output.stderr).unwrap_or_default();
        Err(ScpError::Scp(stderr))
    } else {
        Ok(())
    }
}
