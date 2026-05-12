#!/usr/bin/env python
"""Deploy Lasserre SDP d=18 run on AWS x1e.32xlarge spot + auto-resume.

Usage:
    # One-off full deploy (launch + install + run)
    python deploy_d18_aws_spot.py --ami ami-0abcdef1234567890

    # Check status of existing run
    python deploy_d18_aws_spot.py --status

    # Tail the remote log
    python deploy_d18_aws_spot.py --log

    # SSH into the live instance manually
    python deploy_d18_aws_spot.py --ssh

    # Fetch checkpoint + final result to local data/
    python deploy_d18_aws_spot.py --fetch

    # Nuke everything (instance + EBS + SG) — billing stops.
    python deploy_d18_aws_spot.py --teardown

Flow:
    1. On first launch, create a security group, a 10 GB gp3 EBS volume
       (persistent — survives instance preemption), and submit a one-
       shot spot request for x1e.32xlarge.
    2. Once the instance comes up, attach the EBS at /data, install
       MOSEK + deps + upload code, upload MOSEK license.
    3. Start the run in a detached nohup process with --checkpoint
       /data/ckpt.json.  The run self-checkpoints after each CG round
       and on AWS spot-termination warning.
    4. Local watcher polls the instance.  When the instance dies:
         a. If ckpt says converged=True — fetch results, teardown.
         b. Else — create a NEW spot instance, re-attach the SAME EBS
            (so /data/ckpt.json is already present), bootstrap, run
            with resume.  Repeat up to --max-respawns times.

State file `deploy_d18_state.json` in local `data/` remembers resource
IDs across invocations.

See AWS_SETUP.md for first-time account prep.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
STATE_FILE = PROJECT_ROOT / 'data' / 'deploy_d18_state.json'
MOSEK_LIC_LOCAL = Path.home() / 'mosek' / 'mosek.lic'

DEFAULT_REGION = 'us-east-1'
DEFAULT_INSTANCE_TYPE = 'x1e.32xlarge'
DEFAULT_KEY_NAME = 'sidon-d18'
DEFAULT_SG_NAME = 'sidon-d18-sg'
DEFAULT_EBS_GB = 20      # room for code + ckpt + logs
REMOTE_USER = 'ubuntu'
REMOTE_WORKDIR = '/data/sidon'
REMOTE_CKPT = '/data/ckpt.json'
REMOTE_LOG = '/data/run.log'

FILES_TO_SYNC = [
    'tests/lasserre_fusion_z2cg.py',
    'tests/lasserre_scalable.py',
    'tests/lasserre_fusion.py',
    'tests/lasserre_mosek_tuned.py',
    'lasserre/__init__.py',
    'lasserre/core.py',
    'lasserre/precompute.py',
    'lasserre/cliques.py',
    'lasserre/solvers.py',
    'lasserre/z2_elim.py',
    'lasserre/z2_blockdiag.py',
    'lasserre/z2_symmetry.py',
    'lasserre/dual_sdp.py',
]

# Default run args — user can override via --run-args.
DEFAULT_RUN_ARGS = (
    '--d 18 --order 3 --threads 96 '
    '--cg-rounds 12 --cg-add 20 --bisect 5 '
    '--no-upper-loc --tol 1e-6 '
    '--checkpoint /data/ckpt.json '
    '--json /data/result.json'
)


# ---------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------
# SSH helpers — use the CLI, not paramiko (simpler)
# ---------------------------------------------------------------------

def ssh_opts() -> list:
    return [
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'LogLevel=ERROR',
        '-o', 'ConnectTimeout=20',
        '-i', str(Path.home() / '.ssh' / f'{DEFAULT_KEY_NAME}.pem'),
    ]


def ssh(host: str, cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['ssh', *ssh_opts(), f'{REMOTE_USER}@{host}', cmd],
        capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL)


def scp_to(host: str, local: str, remote: str,
           timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['scp', *ssh_opts(), local, f'{REMOTE_USER}@{host}:{remote}'],
        capture_output=True, text=True, timeout=timeout)


def scp_from(host: str, remote: str, local: str,
             timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['scp', *ssh_opts(), f'{REMOTE_USER}@{host}:{remote}', local],
        capture_output=True, text=True, timeout=timeout)


def wait_for_ssh(host: str, timeout_s: int = 300) -> bool:
    """Poll until the instance accepts SSH.  Spot instances can take
    2-10 min to boot + get SSH daemon ready."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = ssh(host, 'echo ok', timeout=15)
        if r.returncode == 0 and 'ok' in r.stdout:
            return True
        time.sleep(15)
    return False


# ---------------------------------------------------------------------
# AWS resource management
# ---------------------------------------------------------------------

def get_default_vpc_sg(ec2, region: str) -> str:
    """Find or create the sidon-d18-sg security group allowing SSH."""
    resp = ec2.describe_security_groups(
        Filters=[{'Name': 'group-name', 'Values': [DEFAULT_SG_NAME]}])
    if resp['SecurityGroups']:
        return resp['SecurityGroups'][0]['GroupId']
    # Create
    vpcs = ec2.describe_vpcs(
        Filters=[{'Name': 'isDefault', 'Values': ['true']}])
    if not vpcs['Vpcs']:
        raise RuntimeError(
            'No default VPC in region.  Create one in AWS Console → VPC.')
    vpc_id = vpcs['Vpcs'][0]['VpcId']
    resp = ec2.create_security_group(
        GroupName=DEFAULT_SG_NAME,
        Description='SSH ingress for sidon d=18 deploy',
        VpcId=vpc_id)
    sg_id = resp['GroupId']
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22,
            'IpRanges': [{'CidrIp': '0.0.0.0/0',
                          'Description': 'SSH from anywhere'}],
        }])
    print(f'  created security group {sg_id}')
    return sg_id


def pick_subnet(ec2) -> str:
    """Pick any subnet in the default VPC."""
    vpcs = ec2.describe_vpcs(
        Filters=[{'Name': 'isDefault', 'Values': ['true']}])
    vpc_id = vpcs['Vpcs'][0]['VpcId']
    subnets = ec2.describe_subnets(
        Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
    if not subnets['Subnets']:
        raise RuntimeError('No subnets in default VPC.')
    return subnets['Subnets'][0]['SubnetId']


def get_or_create_ebs(ec2, state: Dict[str, Any], az: str,
                      size_gb: int) -> str:
    """Reuse an existing EBS volume from state if present; else create."""
    vol_id = state.get('ebs_volume_id')
    if vol_id:
        try:
            r = ec2.describe_volumes(VolumeIds=[vol_id])
            if r['Volumes'] and r['Volumes'][0]['State'] != 'deleted':
                print(f'  reusing EBS volume {vol_id} ({size_gb} GiB)')
                return vol_id
        except ClientError:
            pass
    # Create new
    r = ec2.create_volume(
        AvailabilityZone=az,
        Size=size_gb,
        VolumeType='gp3',
        TagSpecifications=[{
            'ResourceType': 'volume',
            'Tags': [{'Key': 'Name', 'Value': 'sidon-d18-ckpt'}]}])
    vol_id = r['VolumeId']
    print(f'  created EBS volume {vol_id} ({size_gb} GiB) in {az}')
    # Wait for available
    for _ in range(30):
        s = ec2.describe_volumes(VolumeIds=[vol_id])['Volumes'][0]['State']
        if s == 'available':
            break
        time.sleep(5)
    state['ebs_volume_id'] = vol_id
    state['ebs_az'] = az
    save_state(state)
    return vol_id


def submit_spot(ec2, ami: str, sg_id: str, subnet_id: str,
                instance_type: str) -> str:
    """Submit a ONE-TIME spot request (not persistent — we handle
    respawns ourselves so we can re-sync code on each new instance).
    """
    resp = ec2.request_spot_instances(
        InstanceCount=1,
        Type='one-time',
        LaunchSpecification={
            'ImageId': ami,
            'InstanceType': instance_type,
            'KeyName': DEFAULT_KEY_NAME,
            'SecurityGroupIds': [sg_id],
            'SubnetId': subnet_id,
            # Note: we do NOT attach EBS here because AttachVolume
            # happens post-launch (EBS is reused across respawns).
        })
    req_id = resp['SpotInstanceRequests'][0]['SpotInstanceRequestId']
    print(f'  spot request {req_id} submitted')
    return req_id


def wait_for_spot_fulfillment(ec2, req_id: str,
                              timeout_s: int = 1200) -> str:
    """Block until the spot request is fulfilled and return the
    instance id.  Raises if capacity-constrained longer than timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = ec2.describe_spot_instance_requests(
            SpotInstanceRequestIds=[req_id])
        req = r['SpotInstanceRequests'][0]
        status = req['Status']['Code']
        state = req['State']
        if state == 'active' and req.get('InstanceId'):
            return req['InstanceId']
        if state in ('closed', 'cancelled', 'failed'):
            raise RuntimeError(
                f'spot request {req_id} ended in state={state} '
                f'status={status}: {req["Status"].get("Message")}')
        elapsed = int(time.time() - t0)
        print(f'  spot request {req_id}: state={state} status={status} '
              f'({elapsed}s)', flush=True)
        time.sleep(20)
    raise RuntimeError(f'spot request {req_id} not fulfilled in '
                       f'{timeout_s}s')


def wait_for_instance_running(ec2, instance_id: str,
                              timeout_s: int = 600) -> Dict[str, Any]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = ec2.describe_instances(InstanceIds=[instance_id])
        inst = r['Reservations'][0]['Instances'][0]
        state = inst['State']['Name']
        if state == 'running':
            return inst
        if state in ('shutting-down', 'terminated', 'stopped'):
            raise RuntimeError(f'instance {instance_id} state={state}')
        time.sleep(10)
    raise RuntimeError(f'instance {instance_id} never reached running')


def attach_ebs(ec2, volume_id: str, instance_id: str,
               device: str = '/dev/sdf') -> None:
    # Detach from any previous attachment first (left over from a past
    # preempted instance).
    try:
        vol = ec2.describe_volumes(VolumeIds=[volume_id])['Volumes'][0]
        for att in vol.get('Attachments', []):
            if att['InstanceId'] != instance_id:
                print(f'  detaching {volume_id} from old instance '
                      f'{att["InstanceId"]}')
                ec2.detach_volume(VolumeId=volume_id, Force=True)
                for _ in range(30):
                    s = ec2.describe_volumes(VolumeIds=[volume_id])[
                        'Volumes'][0]['State']
                    if s == 'available':
                        break
                    time.sleep(5)
    except ClientError:
        pass
    ec2.attach_volume(
        VolumeId=volume_id, InstanceId=instance_id, Device=device)
    print(f'  attaching {volume_id} to {instance_id} as {device}')
    for _ in range(30):
        vol = ec2.describe_volumes(VolumeIds=[volume_id])['Volumes'][0]
        atts = vol.get('Attachments', [])
        if atts and atts[0]['State'] == 'attached':
            print(f'  attached.')
            return
        time.sleep(5)
    raise RuntimeError(f'EBS {volume_id} never attached')


def teardown_all(ec2, state: Dict[str, Any]) -> None:
    """Terminate instance, delete EBS, cancel spot request, delete SG."""
    for inst_id in state.get('instance_ids', []):
        try:
            ec2.terminate_instances(InstanceIds=[inst_id])
            print(f'  terminated {inst_id}')
        except ClientError as e:
            print(f'  {inst_id}: {e}')
    # Wait for termination, then delete EBS
    if state.get('ebs_volume_id'):
        for _ in range(60):
            try:
                vol = ec2.describe_volumes(
                    VolumeIds=[state['ebs_volume_id']])['Volumes'][0]
                if vol['State'] in ('available',):
                    break
                time.sleep(5)
            except ClientError:
                break
        try:
            ec2.delete_volume(VolumeId=state['ebs_volume_id'])
            print(f'  deleted EBS {state["ebs_volume_id"]}')
        except ClientError as e:
            print(f'  EBS delete: {e}')
    for req_id in state.get('spot_request_ids', []):
        try:
            ec2.cancel_spot_instance_requests(
                SpotInstanceRequestIds=[req_id])
        except ClientError:
            pass
    # Leave the security group — it's free and reusable.


# ---------------------------------------------------------------------
# Bootstrap + run launch on the remote instance
# ---------------------------------------------------------------------

BOOTSTRAP_SCRIPT = r"""#!/bin/bash
set -euo pipefail
echo "=== bootstrap starting $(date -u) ==="

# Mount the EBS at /data (first time only: format xfs).
DEV=$(ls /dev/nvme1n1 2>/dev/null || ls /dev/xvdf 2>/dev/null || ls /dev/sdf 2>/dev/null || true)
if [ -z "$DEV" ]; then
    echo "EBS device not found; listing:"
    ls -la /dev/nvme* /dev/xvd* /dev/sd* 2>&1 | head
    exit 1
fi
echo "EBS device: $DEV"

# Check if already formatted
if ! sudo file -s "$DEV" | grep -q XFS; then
    echo "formatting $DEV as xfs"
    sudo mkfs.xfs "$DEV"
fi
sudo mkdir -p /data
if ! mountpoint -q /data; then
    sudo mount "$DEV" /data
fi
sudo chown -R ubuntu:ubuntu /data
echo "/data mounted: $(df -h /data | tail -1)"

# Install system deps (idempotent)
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3 python3-pip python3-venv htop
pip3 install --quiet --user mosek numpy scipy requests

# Ensure mosek license dir
mkdir -p ~/mosek

# Code dir on the EBS so it persists across respawns
mkdir -p /data/sidon/tests /data/sidon/lasserre

echo "=== bootstrap done $(date -u) ==="
"""


LAUNCH_SCRIPT = r"""#!/bin/bash
set -u
export PYTHONPATH=/data/sidon:/data/sidon/tests
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd /data/sidon

# Launch detached; tail the log to terminal when `--log` is used.
echo "=== run start $(date -u) ==="
echo "cmd: python3 tests/lasserre_fusion_z2cg.py RUN_ARGS_PLACEHOLDER"
nohup python3 -u tests/lasserre_fusion_z2cg.py RUN_ARGS_PLACEHOLDER \
    > /data/run.log 2>&1 &
echo "PID=$!"
"""


def bootstrap_instance(host: str) -> None:
    print(f'\n=== bootstrap on {host} ===', flush=True)
    # Upload the bootstrap script + run it
    bootstrap_b64 = base64.b64encode(
        BOOTSTRAP_SCRIPT.encode()).decode()
    r = ssh(host,
            f'echo "{bootstrap_b64}" | base64 -d | bash',
            timeout=300)
    print(r.stdout)
    if r.returncode != 0:
        print('BOOTSTRAP FAILED stderr:', r.stderr)
        raise RuntimeError('bootstrap failed')

    # Upload MOSEK license
    if not MOSEK_LIC_LOCAL.exists():
        raise RuntimeError(
            f'No MOSEK license at {MOSEK_LIC_LOCAL}.  Please '
            f'copy one there before deploying.')
    r = scp_to(host, str(MOSEK_LIC_LOCAL), '~/mosek/mosek.lic')
    if r.returncode != 0:
        raise RuntimeError(f'MOSEK license upload failed: {r.stderr}')
    print('  uploaded MOSEK license')

    # Sync code to /data/sidon (persistent across respawns)
    for f in FILES_TO_SYNC:
        local = PROJECT_ROOT / f
        if not local.exists():
            print(f'  SKIP (missing): {f}')
            continue
        remote = f'/data/sidon/{f}'
        # Ensure parent dir
        parent = os.path.dirname(remote)
        ssh(host, f'mkdir -p {parent}', timeout=30)
        r = scp_to(host, str(local), remote)
        if r.returncode != 0:
            print(f'  FAIL {f}: {r.stderr}')
        else:
            print(f'  uploaded {f}')

    # Quick MOSEK import sanity
    r = ssh(host,
            'python3 -c "import mosek; print(\'mosek\', mosek.Env.getversion())"',
            timeout=60)
    print('  ', r.stdout.strip())


def launch_run(host: str, run_args: str) -> None:
    script = LAUNCH_SCRIPT.replace('RUN_ARGS_PLACEHOLDER', run_args)
    b64 = base64.b64encode(script.encode()).decode()
    r = ssh(host, f'echo "{b64}" | base64 -d | bash', timeout=60)
    print(r.stdout)
    if r.returncode != 0:
        print('stderr:', r.stderr)


def tail_log(host: str, n: int = 50) -> None:
    r = ssh(host,
            f'ls -la {REMOTE_LOG} 2>&1 | head; echo ----; '
            f'tail -n {n} {REMOTE_LOG} 2>&1; echo ----; '
            f'pgrep -af lasserre_fusion_z2cg 2>&1 | head',
            timeout=30)
    print(r.stdout)


def check_converged(host: str) -> bool:
    """Read ckpt.json on the instance and return True iff converged."""
    r = ssh(host, f'cat {REMOTE_CKPT} 2>/dev/null', timeout=30)
    if r.returncode != 0 or not r.stdout.strip():
        return False
    try:
        ckpt = json.loads(r.stdout)
        return bool(ckpt.get('converged'))
    except Exception:
        return False


def fetch_results(host: str) -> None:
    """Pull ckpt, log, and result json down to local data/."""
    local_dir = PROJECT_ROOT / 'data' / 'd18_aws'
    local_dir.mkdir(parents=True, exist_ok=True)
    for remote in (REMOTE_CKPT, REMOTE_CKPT + '.y.npz', REMOTE_LOG,
                   '/data/result.json'):
        local = local_dir / os.path.basename(remote)
        r = scp_from(host, remote, str(local))
        status = 'ok' if r.returncode == 0 else f'FAIL: {r.stderr.strip()}'
        print(f'  {remote} -> {local}  {status}')


# ---------------------------------------------------------------------
# Watcher loop (handles spot preemption → respawn)
# ---------------------------------------------------------------------

def monitor_and_respawn(ec2, state: Dict[str, Any],
                        run_args: str, ami: str,
                        instance_type: str,
                        max_respawns: int = 5,
                        poll_s: int = 60) -> None:
    """Poll the instance until it finishes or dies.
    On death, re-launch a new spot instance attached to the same EBS
    (which holds the checkpoint), bootstrap, and resume.
    """
    respawns = 0
    while True:
        inst_id = state['instance_ids'][-1]
        host = state['instance_hosts'][-1]

        # Poll: is the instance alive?  is the run converged?
        try:
            r = ec2.describe_instances(InstanceIds=[inst_id])
            inst_state = r['Reservations'][0]['Instances'][0]['State']['Name']
        except ClientError:
            inst_state = 'unknown'

        if inst_state in ('terminated', 'shutting-down',
                          'stopped', 'stopping'):
            print(f'\n  ! instance {inst_id} died (state={inst_state})',
                  flush=True)
            # Was it converged?
            # Can't SSH after termination; reconstructing from the
            # locally-cached last check is our best signal.  If the
            # last tail_log showed converged=True we'd have already
            # exited this loop.  Assume preemption and respawn.
            if respawns >= max_respawns:
                print(f'  max respawns ({max_respawns}) reached; '
                      f'giving up.', flush=True)
                return
            respawns += 1
            print(f'\n=== RESPAWN {respawns}/{max_respawns} ===',
                  flush=True)
            provision_instance(ec2, state, ami, instance_type)
            save_state(state)
            # Bootstrap + resume (ckpt is on the EBS we just reattached)
            host = state['instance_hosts'][-1]
            if not wait_for_ssh(host):
                print(f'  new instance {host} did not come up; '
                      f'will respawn again.', flush=True)
                state['instance_ids'].append(state['instance_ids'][-1])
                continue
            bootstrap_instance(host)
            launch_run(host, run_args)
            continue

        # Check run status
        r = ssh(host, f'pgrep -af lasserre_fusion_z2cg | head', timeout=20)
        if r.returncode != 0 or 'lasserre_fusion_z2cg' not in r.stdout:
            # Process is gone.  Check convergence.
            converged = check_converged(host)
            if converged:
                print(f'\n=== converged! fetching results ===',
                      flush=True)
                fetch_results(host)
                return
            else:
                print(f'  ! run process gone but NOT converged on {host}',
                      flush=True)
                # Try relaunching the process (same instance)
                launch_run(host, run_args)
                time.sleep(30)
                continue

        # Alive and running.  Show a one-line status.
        r = ssh(host, 'free -g | head -2 | tail -1 ; '
                      'cat /data/ckpt.json 2>/dev/null | python3 -c "'
                      'import sys,json; '
                      'r=json.load(sys.stdin); '
                      'print(\'active=\', len(r[\\\"active_windows\\\"]), '
                      '\'lo=\', round(r[\\\"lo\\\"], 6), '
                      '\'hi=\', round(r[\\\"hi\\\"], 6), '
                      '\'best=\', round(r[\\\"best_lb\\\"], 6), '
                      '\'conv=\', r[\\\"converged\\\"])"',
                timeout=20)
        print(f'  [{time.strftime("%H:%M:%S")}] alive: {r.stdout.strip()}',
              flush=True)
        time.sleep(poll_s)


def provision_instance(ec2, state: Dict[str, Any], ami: str,
                       instance_type: str) -> None:
    """Submit spot, wait for running, attach EBS.  Updates state
    in-place."""
    sg_id = state.setdefault('sg_id',
                              get_default_vpc_sg(ec2, state['region']))
    subnet_id = state.setdefault('subnet_id', pick_subnet(ec2))

    # Fetch subnet's AZ to match EBS AZ
    r = ec2.describe_subnets(SubnetIds=[subnet_id])
    az = r['Subnets'][0]['AvailabilityZone']
    vol_id = get_or_create_ebs(ec2, state, az, DEFAULT_EBS_GB)

    req_id = submit_spot(ec2, ami, sg_id, subnet_id, instance_type)
    state.setdefault('spot_request_ids', []).append(req_id)
    save_state(state)

    inst_id = wait_for_spot_fulfillment(ec2, req_id)
    inst = wait_for_instance_running(ec2, inst_id)
    host = inst['PublicIpAddress']
    state.setdefault('instance_ids', []).append(inst_id)
    state.setdefault('instance_hosts', []).append(host)

    attach_ebs(ec2, vol_id, inst_id)
    print(f'  instance {inst_id} @ {host} live with EBS {vol_id} '
          f'attached', flush=True)
    # Wait a bit so device is visible
    time.sleep(15)


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ami', type=str,
                    help='Ubuntu 22.04 AMI ID.  Required on first run.')
    ap.add_argument('--region', type=str, default=DEFAULT_REGION)
    ap.add_argument('--instance-type', type=str,
                    default=DEFAULT_INSTANCE_TYPE)
    ap.add_argument('--run-args', type=str, default=DEFAULT_RUN_ARGS,
                    help='Extra args to pass to lasserre_fusion_z2cg.py.')
    ap.add_argument('--max-respawns', type=int, default=5)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--ssh', action='store_true')
    ap.add_argument('--teardown', action='store_true')
    args = ap.parse_args()

    state = load_state()
    if not state.get('region'):
        state['region'] = args.region
    ec2 = boto3.client('ec2', region_name=state['region'])

    if args.teardown:
        teardown_all(ec2, state)
        # Reset the state to allow a fresh launch next time.
        keep = {'region': state['region']}
        save_state(keep)
        return 0

    if args.status:
        if not state.get('instance_ids'):
            print('No deploy in progress.')
            return 0
        inst_id = state['instance_ids'][-1]
        r = ec2.describe_instances(InstanceIds=[inst_id])
        inst = r['Reservations'][0]['Instances'][0]
        print(f'instance: {inst_id}  state: {inst["State"]["Name"]}  '
              f'ip: {inst.get("PublicIpAddress", "-")}')
        if inst['State']['Name'] == 'running':
            tail_log(inst['PublicIpAddress'], n=20)
        return 0

    if args.log:
        if not state.get('instance_hosts'):
            print('No deploy in progress.')
            return 1
        tail_log(state['instance_hosts'][-1], n=100)
        return 0

    if args.fetch:
        if not state.get('instance_hosts'):
            print('No deploy in progress.')
            return 1
        fetch_results(state['instance_hosts'][-1])
        return 0

    if args.ssh:
        if not state.get('instance_hosts'):
            print('No deploy in progress.')
            return 1
        host = state['instance_hosts'][-1]
        key = Path.home() / '.ssh' / f'{DEFAULT_KEY_NAME}.pem'
        print(f'ssh -i {key} {REMOTE_USER}@{host}')
        return 0

    # Full deploy
    if not args.ami:
        print('error: --ami is required for a fresh deploy. '
              'See AWS_SETUP.md step 6.', file=sys.stderr)
        return 2
    if not state.get('instance_ids'):
        print(f'=== deploying x1e.32xlarge spot in {state["region"]} ===')
        provision_instance(ec2, state, args.ami, args.instance_type)
        save_state(state)
        host = state['instance_hosts'][-1]
        print(f'\n  waiting for SSH on {host}...')
        if not wait_for_ssh(host):
            print(f'SSH never came up.  Check with --status.')
            return 1
        bootstrap_instance(host)
        launch_run(host, args.run_args)
        print(f'\n  run started.  tail with: '
              f'python deploy_d18_aws_spot.py --log')

    monitor_and_respawn(
        ec2, state, args.run_args, args.ami, args.instance_type,
        max_respawns=args.max_respawns)

    print('\n=== done.  run `--teardown` when you want to clean up. ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())
