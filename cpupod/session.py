"""Session state persistence and orchestration for CPU pods."""
import json
import sys

from .config import SESSION_FILE, PROJECT_ROOT
from .budget import BudgetTracker


class Session:
    """Manages the full CPU pod session lifecycle."""

    def __init__(self):
        self.pod_id = None
        self.ssh_host = None
        self.ssh_port = None
        self.budget = BudgetTracker()
        self._load()

    def _load(self):
        if SESSION_FILE.exists():
            try:
                data = json.loads(SESSION_FILE.read_text())
                self.pod_id = data.get("pod_id")
                self.ssh_host = data.get("ssh_host")
                self.ssh_port = data.get("ssh_port")
                if "start_time" in data:
                    self.budget.start_time = data["start_time"]
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        data = {
            "pod_id": self.pod_id,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "start_time": self.budget.start_time,
        }
        tmp = SESSION_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(SESSION_FILE)

    def _clear(self):
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
        self.pod_id = None
        self.ssh_host = None
        self.ssh_port = None

    def _require_active(self):
        if not self.pod_id or not self.ssh_host:
            print("No active session. Run 'start' first.")
            sys.exit(1)

    def start(self):
        """Create CPU pod, sync code, install deps, verify CPU."""
        try:
            from .pod_manager import create_pod, get_pod_status
        except ImportError:
            print("Error: 'runpod' package not installed. Run: pip install runpod")
            sys.exit(1)
        from .sync import sync_code
        from .remote import install_deps, verify_cpu

        if self.pod_id:
            status = get_pod_status(self.pod_id)
            if status and status.get("desiredStatus") == "RUNNING":
                print(f"Session already active: pod {self.pod_id}")
                print(f"  SSH: ssh -p {self.ssh_port} root@{self.ssh_host}")
                print(self.budget.status_line())
                return

        # Create pod
        info = create_pod()
        self.pod_id = info["pod_id"]
        self.ssh_host = info["ssh_host"]
        self.ssh_port = info["ssh_port"]

        # Start budget tracking
        self.budget.start()
        self._save()

        print(f"\nSSH: ssh -p {self.ssh_port} root@{self.ssh_host}")
        print(self.budget.status_line())

        # Sync, install deps, verify — no CUDA build needed
        try:
            sync_code(self.ssh_host, self.ssh_port)
            install_deps(self.ssh_host, self.ssh_port)
            verify_cpu(self.ssh_host, self.ssh_port)
        except Exception as e:
            print(f"\nSetup failed: {e}")
            print("Pod is still running (costing money). Session saved.")
            print("Fix the issue and retry with 'sync', or 'teardown' to stop.")
            raise

        print("\nPod ready! Use 'sync' after local edits, 'run' to execute.")

    def sync(self):
        """Re-sync code after local edits."""
        from .sync import sync_code

        self._require_active()
        ok, msg = self.budget.check()
        print(msg)
        if not ok:
            sys.exit(1)
        sync_code(self.ssh_host, self.ssh_port)

    def run(self, script=None, args="", auto_teardown=False):
        """Run a CPU job on the pod."""
        from .remote import run_script

        self._require_active()
        ok, msg = self.budget.check()
        print(msg)
        if not ok:
            print("Budget exceeded. Tear down the pod.")
            sys.exit(1)

        timeout = int(self.budget.max_remaining_seconds())
        if timeout < 60:
            print(f"Only {timeout}s of budget remaining. Tear down the pod.")
            sys.exit(1)

        log_dir = str(PROJECT_ROOT / "data")
        kwargs = {"timeout": timeout, "log_dir": log_dir}
        if script:
            kwargs["script"] = script
        if args:
            kwargs["args"] = args

        rc = run_script(self.ssh_host, self.ssh_port, **kwargs)

        print(self.budget.status_line())

        if auto_teardown:
            if rc == 255:
                print("\nSSH connection dropped — job may still be running.")
                print("Skipping auto-teardown. Use 'cpupod status' to check.")
            else:
                print("\n=== AUTO-TEARDOWN ===")
                self.teardown()

        return rc

    def status(self):
        """Show pod state, job state, and budget."""
        if not self.pod_id:
            print("No active session.")
            return

        print(f"Pod ID: {self.pod_id}")
        print(f"SSH: ssh -p {self.ssh_port} root@{self.ssh_host}")

        try:
            from .pod_manager import get_pod_status
            pod_status = get_pod_status(self.pod_id)
            if pod_status:
                desired = pod_status.get("desiredStatus", "UNKNOWN")
                uptime = pod_status.get("uptimeSeconds", 0)
                print(f"Pod: {desired} (uptime: {uptime}s)")
            else:
                print("Pod: UNKNOWN (may have been terminated)")
        except ImportError:
            print("Pod: (runpod SDK not installed)")

        try:
            from .remote import check_job_status
            job = check_job_status(self.ssh_host, self.ssh_port)
            print(f"Job: {job}")
        except Exception:
            print("Job: (could not check — SSH may be down)")

        print(self.budget.status_line())

    def launch(self, script=None, args="", auto_teardown=False):
        """Launch a CPU job in a detached tmux session."""
        from .remote import launch_script, check_job_status

        self._require_active()
        ok, msg = self.budget.check()
        print(msg)
        if not ok:
            print("Budget exceeded. Tear down the pod.")
            sys.exit(1)

        script = script or "cloninger-steinerberger/cpu/run_cascade.py"

        status = check_job_status(self.ssh_host, self.ssh_port)
        if status == "RUNNING":
            print("A job is already running on the pod!")
            print("Use 'logs' to check it, or 'logs -f' to follow live.")
            sys.exit(1)

        api_key = None
        if auto_teardown:
            from .config import RUNPOD_API_KEY
            if not RUNPOD_API_KEY or not self.pod_id:
                print("WARNING: --auto-teardown requires RUNPOD_API_KEY.")
                print("Proceeding WITHOUT auto-teardown.")
                auto_teardown = False
            else:
                api_key = RUNPOD_API_KEY

        if auto_teardown:
            print(f"Launching detached (AUTO-TEARDOWN): {script} {args}")
        else:
            print(f"Launching detached: {script} {args}")
        rc = launch_script(self.ssh_host, self.ssh_port, script=script,
                           args=args, auto_teardown=auto_teardown,
                           api_key=api_key, pod_id=self.pod_id)
        if rc != 0:
            print(f"Failed to launch (exit code {rc})")
            sys.exit(1)

        print(f"\nJob launched in background on pod.")
        print(f"You can now close your laptop. The job keeps running.")
        if auto_teardown:
            print(f"\n  *** Pod will SELF-TERMINATE when the job finishes ***")
            print(f"  *** Fetch results BEFORE that, or they will be lost ***")
        print(f"")
        print(f"  cpupod logs        # see last 80 lines")
        print(f"  cpupod logs -f     # follow live (Ctrl-C to detach)")
        print(f"  cpupod status      # check pod + job state")
        print(f"  cpupod fetch       # pull results to local data/")
        print(f"  cpupod teardown    # stop pod + collect results")

    def logs(self, follow=False, lines=80):
        """Show output from a launched job."""
        from .remote import tail_remote_log, check_job_status

        self._require_active()

        status = check_job_status(self.ssh_host, self.ssh_port)
        print(f"Job status: {status}")
        print(f"{'=' * 60}")
        tail_remote_log(self.ssh_host, self.ssh_port,
                        follow=follow, lines=lines)

    def fetch(self):
        """Pull results from the pod to local data/ directory."""
        from .sync import collect_results

        self._require_active()
        collect_results(self.ssh_host, self.ssh_port)

    def ssh_command(self):
        """Print the SSH command for manual use."""
        self._require_active()
        print(f"ssh -p {self.ssh_port} -i ~/.ssh/id_ed25519 "
              f"-o StrictHostKeyChecking=no root@{self.ssh_host}")

    def teardown(self):
        """Collect results, destroy pod, report cost."""
        if not self.pod_id:
            print("No active session.")
            return

        from .pod_manager import terminate_pod
        from .sync import collect_results

        print("=== TEARDOWN ===")

        if self.ssh_host and self.ssh_port:
            try:
                from .remote import check_job_status
                job = check_job_status(self.ssh_host, self.ssh_port)
                if job == "RUNNING":
                    print("WARNING: A job is still running!")
                    print("Results may be incomplete.")
                    resp = input("Proceed with teardown anyway? [y/N] ")
                    if resp.lower() != "y":
                        print("Teardown cancelled.")
                        return
            except Exception:
                pass

        if self.ssh_host and self.ssh_port:
            try:
                collect_results(self.ssh_host, self.ssh_port)
            except Exception as e:
                print(f"Warning: could not collect results: {e}")

        cost = self.budget.current_cost()
        hours = self.budget.elapsed_hours()
        print(f"\nSession cost: ${cost:.2f} ({hours:.1f}h)")

        try:
            terminate_pod(self.pod_id)
        except Exception as e:
            print(f"Warning: termination error: {e}")
            print("  Manually check: https://www.runpod.io/console/pods")

        self._clear()
        print("Session ended.")

    def cleanup(self):
        """Emergency: terminate ALL pods."""
        try:
            from .pod_manager import terminate_all_pods
        except ImportError:
            print("Error: 'runpod' package not installed.")
            sys.exit(1)

        print("=== EMERGENCY CLEANUP ===")
        terminate_all_pods()
        self._clear()
        print("All pods terminated. Session cleared.")
