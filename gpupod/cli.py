"""CLI entry point for RunPod GPU integration.

Usage:
    python -m gpupod <command> [args]

Commands:
    start            Create pod, sync code, build CUDA, verify GPU
    sync             Re-sync code after local edits
    build            Recompile CUDA kernels on pod
    run [script]     Run GPU job interactively (dies if SSH drops)
    launch [script]  Run GPU job detached (survives SSH disconnect)
    logs [-f]        Show output from launched job (-f to follow live)
    fetch            Pull results from pod to local data/
    status           Show pod state, job state, + budget
    ssh              Print SSH command for manual use
    teardown         Collect results + destroy pod
    cleanup          Emergency: terminate ALL pods

Typical workflow for long runs:
    gpupod start
    gpupod launch run_proof.py --target 1.20 --m 50
    # close laptop, come back later
    gpupod logs
    gpupod fetch
    gpupod teardown
"""
import sys

from .session import Session


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    session = Session()

    if command == "start":
        session.start()

    elif command == "sync":
        session.sync()

    elif command == "build":
        session.build()

    elif command == "run":
        script = sys.argv[2] if len(sys.argv) > 2 else None
        args = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        session.run(script=script, args=args)

    elif command == "launch":
        script = sys.argv[2] if len(sys.argv) > 2 else None
        args = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        session.launch(script=script, args=args)

    elif command == "logs":
        follow = "-f" in sys.argv[2:]
        # Parse -n <lines> if given
        lines = 80
        for i, arg in enumerate(sys.argv[2:], 2):
            if arg == "-n" and i + 1 < len(sys.argv):
                lines = int(sys.argv[i + 1])
        session.logs(follow=follow, lines=lines)

    elif command == "fetch":
        session.fetch()

    elif command == "status":
        session.status()

    elif command == "ssh":
        session.ssh_command()

    elif command == "teardown":
        session.teardown()

    elif command == "cleanup":
        session.cleanup()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
