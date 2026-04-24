import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.orchestrator import SIOrchestrator
from utils.logger import console, log_error
from utils.config import get_config

BANNER = """
╔══════════════════════════════════════════════════════╗
║        SI-RAG  -  Self-Verifying Adaptive RAG        ║
║    Super-Intelligence Orchestration Layer  v1.0      ║
╚══════════════════════════════════════════════════════╝
"""


def interactive_loop(orchestrator: SIOrchestrator) -> None:
    console.print("[dim]Type your question and press Enter. Type 'exit' to quit.[/dim]\n")
    while True:
        try:
            query = input("Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        try:
            response = orchestrator.run(query)
            console.print(
                f"\n[dim]confidence={response.confidence:.3f}  "
                f"retries={response.total_retries}[/dim]\n"
            )
        except Exception as e:
            log_error(str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="SI-RAG query interface")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()

    if args.no_debug:
        get_config().debug.enabled = False

    console.print(BANNER, style="bold blue")

    try:
        orchestrator = SIOrchestrator()
    except Exception as e:
        log_error(f"Failed to initialize: {e}")
        sys.exit(1)

    if args.query:
        try:
            response = orchestrator.run(args.query)
            if not get_config().debug.enabled:
                console.print(f"\n[bold green]Answer:[/bold green] {response.answer}")
                console.print(f"[dim]Confidence: {response.confidence:.3f} | Retries: {response.total_retries}[/dim]")
        except Exception as e:
            log_error(str(e))
            sys.exit(1)
    else:
        interactive_loop(orchestrator)


if __name__ == "__main__":
    main()