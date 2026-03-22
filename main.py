# main.py

import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from src.pipeline import run_pipeline

console = Console()


def load_file(path):
    """Read a text file and return its contents."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            console.print(f"[red]Error: {path} is empty. Please add content.[/red]")
            sys.exit(1)
        return content
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {path}[/red]")
        sys.exit(1)


def display_results(result):
    """Pretty-print the analysis using rich."""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold yellow]⚡ RAG JD ANALYZER — RESULTS[/bold yellow]",
        border_style="yellow"
    ))
    
    # Show retrieved context
    console.print(f"\n[bold cyan]📥 Retrieved {len(result['retrieved_chunks'])} Resume Sections:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Chunk", style="dim", width=60)
    table.add_column("Relevance Score", justify="right")
    
    for chunk, score in result["retrieved_chunks"]:
        preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
        color = "green" if score > 0.7 else "yellow" if score > 0.5 else "red"
        table.add_row(preview, f"[{color}]{score}[/{color}]")
    
    console.print(table)
    
    # Show LLM analysis
    console.print(Panel(
        result["answer"],
        title="[bold green]📊 Analysis Report[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))
    
    console.print(f"\n[dim]Model used: {result['model']} | "
                  f"Resume chunks indexed: {result['num_chunks']}[/dim]\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAG-powered Job Description Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --resume data/resume.txt --jd data/jd.txt
  python main.py --resume data/resume.txt --jd data/jd.txt --model llama3
  python main.py --resume data/resume.txt --jd data/jd.txt --top_k 7
        """
    )
    
    parser.add_argument("--resume", required=True,  help="Path to your resume (.txt)")
    parser.add_argument("--jd",     required=True,  help="Path to job description (.txt)")
    parser.add_argument("--model",  default="mistral", help="Ollama model (default: mistral)")
    parser.add_argument("--top_k",  default=5, type=int, help="Chunks to retrieve (default: 5)")
    
    args = parser.parse_args()
    
    # Welcome banner
    console.print(Panel.fit(
        "[bold]RAG Job Description Analyzer[/bold]\n"
        "[dim]Resume = Knowledge Base | JD = Query[/dim]",
        border_style="blue"
    ))
    
    # Load files
    console.print(f"\n[cyan]Loading files...[/cyan]")
    resume_text = load_file(args.resume)
    jd_text     = load_file(args.jd)
    console.print(f"  Resume: {len(resume_text.split())} words")
    console.print(f"  JD:     {len(jd_text.split())} words")
    
    # Run pipeline
    try:
        result = run_pipeline(
            resume_text=resume_text,
            jd_text=jd_text,
            model=args.model,
            top_k=args.top_k
        )
    except ConnectionError as e:
        console.print(f"\n[red]{e}[/red]")
        sys.exit(1)
    
    # Display results
    display_results(result)


if __name__ == "__main__":
    main()
