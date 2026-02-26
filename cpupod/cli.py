"""CLI entry point for RunPod CPU pod integration.

Usage:
    python -m cpupod <command> [args]

Commands:
    start            Create CPU pod, sync code, install deps
    sync             Re-sync code after local edits
    run [script]     Run CPU job interactively + stream output
    launch [script]  Run CPU job detached (survives SSH disconnect)
    logs [-f]        Show output from launched job (-f to follow live)
    fetch            Pull results from pod to local data/
    status           Show pod state, job state, + budget
    ssh              Print SSH command for manual use
    teardown         Collect results + destroy pod
    cleanup          Emergency: terminate ALL pods

Typical workflow:
    cpupod start
    cpupod launch run_cascade.py --n_half 2 --m 20 --c_target 1.30 --max_levels 999
    # close laptop, come back later
    cpupod logs
    cpupod fetch
    cpupod teardown
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

    elif command == "run":
        auto_teardown = "--auto-teardown" in sys.argv
        rest = [a for a in sys.argv[2:] if a != "--auto-teardown"]
        script = rest[0] if rest else None
        args = " ".join(rest[1:]) if len(rest) > 1 else ""
        session.run(script=script, args=args, auto_teardown=auto_teardown)

    elif command == "launch":
        auto_teardown = "--auto-teardown" in sys.argv
        rest = [a for a in sys.argv[2:] if a != "--auto-teardown"]
        script = rest[0] if rest else None
        args = " ".join(rest[1:]) if len(rest) > 1 else ""
        session.launch(script=script, args=args, auto_teardown=auto_teardown)

    elif command == "logs":
        follow = "-f" in sys.argv[2:]
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
