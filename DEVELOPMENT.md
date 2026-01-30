# Contributor Guide

First off, thank you for considering contributing to `hiperhealth`! We welcome
all contributions, from bug reports to new features and documentation
improvements. This guide provides everything you need to get your development
environment set up and start contributing.

Following these guidelines helps to communicate that you respect the time of the
developers managing and developing this open source project. In return, they
should reciprocate that respect in addressing your issue or assessing patches
and features.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Project Layout](#project-layout)
- [Getting Started: Local Setup](#1-getting-started-local-setup)
- [Development Workflow](#2-development-workflow)
  - [Running the Applications](#running-the-applications)
  - [Database Migrations](#database-migrations)
  - [Code Style & Linting](#code-style--linting)
  - [Running Tests](#running-tests)
- [Types of Contributions](#types-of-contributions)
- [Architectural Overview](#3-architectural-overview)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Submitting Changes](#4-submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project and everyone participating in it is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

---

## Project Layout

This project uses the _src layout_, which means that the package code is located
at `./src/hiperhealth`.

For more information, check the official documentation:
<https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>

---

## 1. Getting Started: Local Setup

This project uses **Conda** to manage environments, **Uv/Setuptools** to manage
dependencies, and **Makim** to streamline development tasks.

Uv/Setuptools is a Python package management tool that simplifies the process of
building and publishing Python packages. It allows us to easily manage
dependencies, virtual environments, and package versions. Uv/Setuptools also
includes features such as dependency resolution, lock files, and publishing to
PyPI.

### Prerequisites

- Python 3.11+
- [Conda](https://github.com/conda-forge/miniforge?tab=readme-ov-file#download)
  installed on your system.

### Installation

1.  **Fork & Clone the Repository:** Start by forking the repository on GitHub,
    then clone your fork locally:

    ```bash
    git clone git@github.com:<your-username>/hiperhealth.git
    cd hiperhealth
    ```

2.  **Create the Development Environment:** Set up a Conda virtual environment
    using the provided `conda/dev.yaml` file:

    ```bash
    conda env create -f conda/dev.yaml
    conda activate hiperhealth
    ```

    Alternatively, if you have `mamba` installed, you can use it for faster
    environment creation:

    ```bash
    mamba env create -f conda/dev.yaml
    mamba activate hiperhealth
    ```

3.  **Install Project Dependencies:** This command installs all required
    packages.

    ```bash
    ./scripts/install-dev.sh
    ```

4.  **Set Up the Database:** Our `makim` task runner simplifies database setup.
    This command runs Alembic to create the `db.sqlite` file and applies all
    migrations.

    ```bash
    makim db.setup
    ```

5.  **(Optional) Set Up API Keys:** Certain tests and features that interact
    with external services (e.g., OpenAI) require API keys. Create a `.env` file
    at `.envs/.env` and add your keys there.
    ```dotenv
    # In .envs/.env
    OPENAI_API_KEY="your-key-here"
    ```

---

## 2. Development Workflow

All common development tasks are managed via `makim` commands defined in
`.makim.yaml`.

### Running the Applications

- **To run the Research Frontend:**

  ```bash
  makim research.frontend
  ```

  The app will be available at `http://127.0.0.1:5173`.

- **To run the Research Backend:**

  ```bash
  makim research.backend
  ```

  The server will listen at `http://127.0.0.1:8000`

- **To run the Research CLI:**
  ```bash
  makim research.cli
  ```

### Database Migrations

The database schema is managed with Alembic. The schema's source of truth is the
set of Pydantic models in `src/hiperhealth/schema/`, which are used to
auto-generate the SQLAlchemy ORM models.

If you modify a Pydantic schema that requires a database change:

1.  **Regenerate the SQLAlchemy Models:**

    ```bash
    makim models.sqla
    ```

2.  **Generate a New Migration Script:** Provide a descriptive message for the
    change.

    ```bash
    makim db.revision -m "A short description of the schema change"
    ```

3.  **Apply the Migration to Your Local Database:**
    ```bash
    makim db.setup
    ```

### Code Style & Linting

We enforce code quality and a consistent style using `pre-commit` hooks,
configured in `.pre-commit-config.yaml`.

- **Install Git Hooks:**

  ```bash
  pre-commit install
  ```

  The hooks will now run automatically on every commit. Usually, the
  verification will only run on the files modified by that commit, but the
  verification can also be triggered for all the files using:

  ```bash
  pre-commit run --all-files
  ```

  If you would like to skip the failing checks and push the code for further
  discussion, use the `--no-verify` option with `git commit`.

- **Run Hooks Manually:** To run the checks on all files at any time:
  ```bash
  makim test.pre-commit
  ```

### Running Tests

Our test suite uses `pytest`.

- **Run All Tests:**

  ```bash
  makim tests.run
  ```

- **Run Tests with Coverage Report:**
  ```bash
  makim tests.coverage
  ```

---

## Types of Contributions

You can contribute to hiperhealth in many ways:

### Report Bugs

Report bugs at [GitHub Issues](../../issues).

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

hiperhealth could always use more documentation, whether as part of the official
hiperhealth docs, in docstrings, or even on the web in blog posts, articles, and
such.

### Submit Feedback

The best way to send feedback is to file an issue at
[GitHub Issues](../../issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are
  welcome :)

---

## 3. Architectural Overview

The `hiperhealth` library follows a "schema-first" approach for its database
models.

1.  **Pydantic Schemas (`src/hiperhealth/schema/`)**: These are the primary
    source of truth. They define the data structures and validation rules for
    our application.
2.  **SQLAlchemy Models (`src/hiperhealth/models/sqla/`)**: These ORM models are
    **auto-generated** from the Pydantic schemas using the
    `scripts/gen_models/gen_sqla.py` script (`makim models.sqla`). **Do not edit
    these files manually.**
3.  **Alembic Migrations (`migrations/`)**: Alembic uses the generated
    SQLAlchemy models to automatically create database migration scripts.

This ensures our application's data layer and database schema are always
perfectly synchronized.

---

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your
   new functionality into a function with a docstring, and add the feature to
   the list in README.md.
3. The pull request should work for Python >= 3.10.
4. Ensure that the commit message follows the
   [Angular Commit Message Conventions](#commit-message-format).

---

## 4. Submitting Changes

1.  Create a new branch for your feature or bugfix.
2.  Make your changes, ensuring you add tests for any new functionality.
3.  Ensure all tests pass and the linter is happy.
4.  Push your branch to your fork and open a Pull Request against the `main`
    branch of the upstream repository.
5.  In your PR description, clearly explain the problem and your solution.
    Include the relevant issue number if applicable.

---

## Release Process

This project uses semantic-release in order to cut a new release based on the
commit message.

### Commit Message Format

**semantic-release** uses the commit messages to determine the consumer impact
of changes in the codebase. Following formalized conventions for commit
messages, **semantic-release** automatically determines the next
[semantic version](https://semver.org) number, generates a changelog, and
publishes the release.

By default, **semantic-release** uses
[Angular Commit Message Conventions](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#-commit-message-format).
The commit message format can be changed with the `preset` or `config` options
of the
[@semantic-release/commit-analyzer](https://github.com/semantic-release/commit-analyzer#options)
and
[@semantic-release/release-notes-generator](https://github.com/semantic-release/release-notes-generator#options)
plugins.

Tools such as [commitizen](https://github.com/commitizen/cz-cli) or
[commitlint](https://github.com/conventional-changelog/commitlint) can be used
to help contributors and enforce valid commit messages.

The table below shows which commit message gets you which release type when
`semantic-release` runs (using the default configuration):

| Commit message                                                 | Release type     |
| -------------------------------------------------------------- | ---------------- |
| `fix(pencil): stop graphite breaking when pressure is applied` | Fix Release      |
| `feat(pencil): add 'graphiteWidth' option`                     | Feature Release  |
| `perf(pencil): remove graphiteWidth option`                    | Chore            |
| `feat(pencil)!: The graphiteWidth option has been removed`     | Breaking Release |

Note: For a breaking change release, use `!` at the end of the message prefix.

Source:
<https://github.com/semantic-release/semantic-release/blob/master/README.md#commit-message-format>

As this project uses the `squash and merge` strategy, ensure to apply the commit
message format to the PR's title.
