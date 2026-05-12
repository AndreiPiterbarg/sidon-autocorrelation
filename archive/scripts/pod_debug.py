"""Debug what paramiko sees from the pod."""
import paramiko
from pathlib import Path

SSH_IDENTITY = str(Path.home() / ".ssh" / "id_ed25519")
print("key:", SSH_IDENTITY)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    hostname="103.196.86.88", port=33611, username="root",
    key_filename=SSH_IDENTITY, timeout=15,
)
print("connected.")
sftp = ssh.open_sftp()
print("cwd:", sftp.getcwd())
try:
    st = sftp.stat("/workspace/sidon-autocorrelation/data/d16_t1p285_32w.log")
    print("stat:", st.st_size, "bytes")
except Exception as e:
    print("stat failed:", e)
try:
    entries = sftp.listdir("/workspace/sidon-autocorrelation/data")
    print("dir contents:", entries)
except Exception as e:
    print("listdir failed:", e)
try:
    with sftp.file("/workspace/sidon-autocorrelation/data/d16_t1p285_32w.log", "rb") as fh:
        data = fh.read()
    print("read", len(data), "bytes")
except Exception as e:
    print("read failed:", e)
sftp.close()
ssh.close()
