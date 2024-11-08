import os
import subprocess
from enum import Enum
from typing import Optional

try:
    import typer
except (ImportError, ):
    os.system("pip install typer")
    import typer

app = typer.Typer()


# Enum for different components
class Component(str, Enum):
    plates = "plates"
    people = "people"
    modification = "modification"
    api = "api"
    all = "all"


# Define environments, you can extend this if needed
class Environment(str, Enum):
    dev = "dev"
    test = "test"
    production = "production"


@app.command()
def venv(venv: Optional[str] = "venv"):
    typer.echo("\nCreating virtual environment 🍇")
    os.system(f"python -m venv .{venv}")
    command = typer.style(f"`source .{venv}/bin/activate`",
                          fg=typer.colors.GREEN,
                          bold=True)
    typer.echo(f"\nActivate with: {command}. Happy coding 😁 \n")


@app.command()
def install():
    typer.echo("\nInstalling packages 🚀")
    os.system("pip install -r requirements.txt")
    typer.echo("\nPackages installed. Have fun 😁 \n")


def run_command(command):
    return subprocess.Popen(command, shell=True)


@app.command("serve")
def serve(
    component: Component,
    env: Optional[Environment] = Environment.dev,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    typer.echo(f"\nRunning | Component: {component} | Environment: {env} 🚀 \n")

    processes = []

    if component == Component.api or component == Component.all:
        typer.echo("\nServing FastAPI backend 🌐...")
        processes.append(
            run_command(f"ENV={env.value} uvicorn src.api.main:app \
                    --reload --host {host} --port {port}"))

    if component == Component.plates or component == Component.all:
        typer.echo("\nServing plate detection 🚗...")
        processes.append(
            run_command(f"ENV={env.value} PYTHONPATH=$(pwd) \
                    python src/core/detect_plates.py"))

    if component == Component.people or component == Component.all:
        typer.echo("\nServing people counting 👥...")
        processes.append(
            run_command(f"ENV={env.value} PYTHONPATH=$(pwd) \
                    python src/core/count_people.py"))
    
    if component == Component.modification or component == Component.all:
        typer.echo("\nServing vehicle modification 🚘...")
        processes.append(
            run_command(f"ENV={env.value} PYTHONPATH=$(pwd) \
                    python src/core/identify_properties.py"))

    for process in processes:
        process.wait()


if __name__ == "__main__":
    app()
