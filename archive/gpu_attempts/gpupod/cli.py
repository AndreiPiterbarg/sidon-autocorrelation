"""CLI entry point for RunPod GPU pod integration.

Usage:
    python -m gpupod <command> [args]

Commands:
    start                Create GPU pod, sync code, build CUDA kernel
    sync                 Re-sync code after local edits
    build                Rebuild CUDA kernel on pod
    upload <file.npy>    Upload checkpoint to pod's data/ directory
    run <level>          Run cascade level interactively (1-4)
    launch <level>       Run cascade level detached (survives disconnect)
    prove                Full proof: generate L0 + run all GPU levels (detached)
    logs [-f]            Show output from launched job
    fetch                Pull results from pod to local data/
    status               Show pod state, job state, + budget
    ssh                  Print SSH command for manual use
    teardown             Collect results + destroy pod
    cleanup              Emergency: terminate ALL pods

Quick proof (recommended):
    gpupod start
    gpupod prove                                  # m=35, c=1.33 (default)
    gpupod prove --m 35 --c_target 1.33           # explicit
    gpupod prove --auto-teardown                  # auto-destroy pod when done
    gpupod logs -f                                # follow progress
    gpupod fetch
    gpupod teardown

Level-by-level workflow:
    gpupod start
    gpupod upload data/checkpoint_L0_survivors.npy
    gpupod run 1          # L0→L1, watch progress
    gpupod run 2          # L1→L2
    gpupod launch 3       # L2→L3 (detached, long run)
    gpupod logs -f        # follow progress
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

    elif command == "upload":
        if len(sys.argv) < 3:
            print("Usage: gpupod upload <file.npy> [remote_name]")
            sys.exit(1)
        local_path = sys.argv[2]
        remote_name = sys.argv[3] if len(sys.argv) > 3 else None
        session.upload(local_path, remote_name)

    elif command == "run":
        if len(sys.argv) < 3:
            print("Usage: gpupod run <level> [--max-survivors N]")
            sys.exit(1)
        auto_teardown = "--auto-teardown" in sys.argv
        level = int(sys.argv[2])
        max_survivors = None
        for i, arg in enumerate(sys.argv):
            if arg == "--max-survivors" and i + 1 < len(sys.argv):
                max_survivors = int(sys.argv[i + 1])
        session.run(level, max_survivors=max_survivors,
                    auto_teardown=auto_teardown)

    elif command == "launch":
        if len(sys.argv) < 3:
            print("Usage: gpupod launch <level> [--auto-teardown] [--m M] [--c_target C]")
            sys.exit(1)
        auto_teardown = "--auto-teardown" in sys.argv
        level = int(sys.argv[2])
        max_survivors = None
        m = None
        c_target = None
        for i, arg in enumerate(sys.argv):
            if arg == "--max-survivors" and i + 1 < len(sys.argv):
                max_survivors = int(sys.argv[i + 1])
            elif arg == "--m" and i + 1 < len(sys.argv):
                m = int(sys.argv[i + 1])
            elif arg == "--c_target" and i + 1 < len(sys.argv):
                c_target = float(sys.argv[i + 1])
        session.launch(level, max_survivors=max_survivors,
                       auto_teardown=auto_teardown, m=m, c_target=c_target)

    elif command == "prove":
        auto_teardown = "--auto-teardown" in sys.argv
        interactive = "--interactive" in sys.argv
        m = 35
        c_target = 1.33
        max_level = 3
        for i, arg in enumerate(sys.argv):
            if arg == "--m" and i + 1 < len(sys.argv):
                m = int(sys.argv[i + 1])
            elif arg == "--c_target" and i + 1 < len(sys.argv):
                c_target = float(sys.argv[i + 1])
            elif arg == "--max_level" and i + 1 < len(sys.argv):
                max_level = int(sys.argv[i + 1])
        session.prove(m=m, c_target=c_target, max_level=max_level,
                      auto_teardown=auto_teardown,
                      detached=not interactive)

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
