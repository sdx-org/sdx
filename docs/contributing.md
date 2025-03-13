# Contributing

In order to be able to contribute, it is important that you understand the
project layout.

This project uses the _src layout_, which means that the package code is located
at `./src/sdx`.

For my information, check the official documentation:
<https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>

In addition, you should know that to build our package we use
[Poetry](https://python-poetry.org/), it's a Python package management tool that
simplifies the process of building and publishing Python packages. It allows us
to easily manage dependencies, virtual environments and package versions. Poetry
also includes features such as dependency resolution, lock files and publishing
to PyPI. Overall, Poetry streamlines the process of managing Python packages,
making it easier for us to create and share our code with others.

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at /issues.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with “bug” and “help
wanted” is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with “enhancement”
and “help wanted” is open to whoever wants to implement it.

### Write Documentation

sdx could always use more documentation, whether as part of the official sdx
docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at /issues.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are
  welcome :)

## Get Started

Ready to contribute? Here’s how to set up `sdx` for local development.

1. Fork the `sdx` repo on GitHub.
2. Clone your fork locally and change to the directory of your project:

```bash
$ git clone git@github.com:your_name_here/sdx.git
$ cd sdx/
```

### Prepare and use virtual environmentWe highly recommend you to use `conda` for managing virtual environment, but

you can use any other one of your preference.

### Install the dependencies

Now, you can already install the dependencies for the project:

````bash
$ poetry install
```### Create a Development Branch

Make a dedicated branch for your bugfix or feature.

```bash
$ git checkout -b name-of-your-bugfix-or-feature
````

### Make Changes Locally

You are now ready to implement your changes or improvements.

### Install and Use Pre-commit Hooks

- `sdx` uses a set of `pre-commit` hooks to improve code quality. The hooks can
  be installed locally using:

```bash
$ pre-commit install
```

This would run the checks every time a `git commit` is executed locally.
Usually, the verification will only run on the files modified by that commit,
but the verification can also be triggered for all the files using:

```bash
$ pre-commit run --all-files
```

If you would like to skip the failing checks and push the code for further
discussion, use the `--no-verify` option with `git commit`.

### Unit Testing with `pytest`

This project uses `pytest` as a testing tool. `pytest` is responsible for
testing the code, whose configuration is available in pyproject.toml.
Additionally, this project also uses `pytest-cov` to calculate the coverage of
these unit tests. For more information, check the section about tests later in
this document.

### Commit your changes and push your branch to GitHub

```bash
$ git add .
$ git commit -m "Your detailed description of your changes.""
$ git push origin name-of-your-bugfix-or-feature
```

- Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your
   new functionality into a function with a docstring, and add the feature to
   the list in README.rst.
3. The pull request should work for Python >= 3.8.

## Running tests locally

The tests can be executed using the `test` dependencies of `sdx` in the
following way:

```bash
$ python -m pytest
```

## Release

This project uses semantic-release in order to cut a new release based on the
commit-message.

### Commit message format

**semantic-release** uses the commit messages to determine the consumer impact
of changes in the codebase. Following formalized conventions for commit
messages, **semantic-release** automatically determines the next
[semantic version](https://semver.org) number, generates a changelog and
publishes the release.

By default, **semantic-release** uses
[Angular Commit Message Conventions](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#-commit-message-format).
The commit message format can be changed with the `preset` or `config` options\_
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

Note: For a breaking change release, uses `!` at the end of the message prefix.

source:
<https://github.com/semantic-release/semantic-release/blob/master/README.md#commit-message-format>

As this project uses the `squash and merge` strategy, ensure to apply the commit
message format to the PR's title.
