"""Module with CLI functions."""

import click

from sdx import __version__


@click.command()
@click.option(
    "--version",
    is_flag=True,
    help="Show the version of the installed sdx tool.",
)
def app(version):
    """Run the application."""
    if version:
        return click.echo(__version__)
    click.echo("You can add more Click commands here.")


if __name__ == "__main__":
    app()
