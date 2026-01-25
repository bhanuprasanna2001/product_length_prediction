"""Centralized logging with Rich formatting."""

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

THEME = Theme({
    "info": "cyan", "warning": "yellow", "error": "bold red",
    "success": "bold green", "metric": "bold magenta", "path": "blue underline",
})

console = Console(theme=THEME)
_logging_configured = False


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None, name: str = "product_length") -> logging.Logger:
    """Configure logging with Rich formatting. Call once at script start."""
    global _logging_configured
    
    logger = logging.getLogger(name)
    if _logging_configured:
        return logger
    
    logger.setLevel(level)
    logger.handlers.clear()
    
    rich_handler = RichHandler(console=console, show_time=True, show_path=False, markup=True, rich_tracebacks=True)
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)
    
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)
    
    logger.propagate = False
    for noisy in ["pytorch_lightning", "wandb", "torch"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    _logging_configured = True
    return logger


def get_logger(name: str = "product_length") -> logging.Logger:
    """Get logger, setting up if needed."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logging(name=name)
    return logger


# Pretty printing
def print_header(title: str, width: int = 60) -> None:
    console.print()
    console.print("=" * width, style="bold cyan")
    console.print(f"[bold white]{title.center(width)}[/]")
    console.print("=" * width, style="bold cyan")


def print_section(title: str, width: int = 60) -> None:
    console.print()
    console.print(f"[bold cyan]{'─' * 20} {title} {'─' * (width - 23 - len(title))}[/]")


def print_config(config_dict: dict, title: str = "Configuration") -> None:
    print_section(title)
    for key, value in config_dict.items():
        if isinstance(value, dict):
            console.print(f"  [cyan]{key}:[/]")
            for k, v in value.items():
                console.print(f"    [white]{k}:[/] {v}")
        else:
            console.print(f"  [cyan]{key}:[/] {value}")


def print_metrics(metrics: dict, title: str = "Metrics") -> None:
    print_section(title)
    for key, value in metrics.items():
        console.print(f"  [metric]{key}:[/] {value:.4f}" if isinstance(value, float) else f"  [metric]{key}:[/] {value}")


def print_success(message: str) -> None:
    console.print(f"[success]✓[/] {message}")


def print_warning(message: str) -> None:
    console.print(f"[warning]⚠[/] {message}")


def print_error(message: str) -> None:
    console.print(f"[error]✗[/] {message}")


def print_info(message: str) -> None:
    console.print(f"[info]ℹ[/] {message}")


def print_progress(current: int, total: int, prefix: str = "") -> None:
    console.print(f"  {prefix}[{current:,}/{total:,}] ({current / total * 100:.1f}%)", end="\r")


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}m" if seconds < 3600 else f"{seconds/3600:.1f}h"


def format_size(bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes < 1024:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024
    return f"{bytes:.1f}TB"
