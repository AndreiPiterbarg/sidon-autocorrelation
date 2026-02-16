"""CLI entry point for RunPod GPU integration.

Usage:
    python -m gpupod <command> [args]

Commands:
    start       Create pod, sync code, build CUDA, verify GPU
    sync        Re-sync code after local edits
    build       Recompile CUDA kernels on pod
    run [script]  Run GPU job (default: cloninger-steinerberger/gpu/solvers.py)
    status      Show pod state + budget
    ssh         Print SSH command for manual use
    teardown    Collect results + destroy pod
    cleanup     Emergency: terminate ALL pods
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
