from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from typing import List, Dict, Any

console = Console()


def log_step(step: str, message: str, color: str = "cyan") -> None:
    console.print(f"[bold {color}][{step}][/bold {color}] {message}")


def log_query(original: str, optimized: str | None = None) -> None:
    console.rule("[bold blue]Query[/bold blue]")
    console.print(f"  [dim]Original :[/dim] {original}")
    if optimized and optimized != original:
        console.print(f"  [green]Optimized:[/green] {optimized}")


def log_retrieved_docs(docs: List[Dict[str, Any]], top_k: int) -> None:
    table = Table(
        title=f"Retrieved Documents (top {top_k})",
        box=box.SIMPLE_HEAD,
        show_lines=False,
        highlight=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=7)
    table.add_column("Source", width=20)
    table.add_column("Snippet", no_wrap=False)

    for i, doc in enumerate(docs[:top_k], 1):
        score = f"{doc.get('score', 0.0):.3f}"
        source = doc.get("source", "—")[:20]
        snippet = doc.get("text", "")[:120].replace("\n", " ") + "..."
        table.add_row(str(i), score, source, snippet)

    console.print(table)


def log_answer(answer: str) -> None:
    console.print(Panel(
        answer,
        title="[bold green]Generated Answer[/bold green]",
        border_style="green"
    ))


def log_confidence(score: float, threshold: float) -> None:
    color = "green" if score >= threshold else "red"
    status = "PASS" if score >= threshold else "FAIL -> retry"
    console.print(
        f"  [bold {color}]Confidence: {score:.3f}[/bold {color}]"
        f"  [dim](threshold {threshold}) -> {status}[/dim]"
    )


def log_retry(attempt: int, strategy: str, reason: str) -> None:
    console.print(
        f"  [bold yellow]Retry {attempt}[/bold yellow] "
        f"strategy=[magenta]{strategy}[/magenta]  "
        f"reason=[dim]{reason}[/dim]"
    )


def log_final_output(
    answer: str,
    confidence: float,
    retries: int,
    citations: List[str]
) -> None:
    console.rule("[bold green]Final Verified Output[/bold green]")
    console.print(Panel(answer, border_style="green"))
    console.print(f"  Confidence : [bold green]{confidence:.3f}[/bold green]")
    console.print(f"  Retries    : {retries}")
    if citations:
        console.print("  Citations  :")
        for c in citations:
            console.print(f"    [dim]-[/dim] {c}")


def log_error(message: str) -> None:
    console.print(f"[bold red][ERROR][/bold red] {message}")