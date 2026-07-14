# nanorod

## Project Overview

This nanorod is a Brownian motion simulation program implemented in Rust and CUDA.

## Physical Model of the Simulation

In this nanorod project, we conduct numerical simulations of the Brownian motion of rod-shaped particles. For detailed information on the physical model, please refer to [rod_simulation.md](docs/rod_simulation.md).

## Computer Architecture

The architecture of the machine used for GPU-based computation is described in [architecture.md](docs/architecture.md). Understanding the machine architecture is important when performing optimizations and similar work.

## Code Quality Practices

- Keep complexity under control through appropriate abstraction, concretization, and use of libraries
  - Remove code and libraries that are no longer needed
- When using a library, consult its documentation and use it correctly
  - Refer to the documentation for how to specify library versions
  - Unless the documentation instructs otherwise, use the latest stable version
- Always attach comments **in Japanese** explaining the meaning of functions, structs, and any other semantically cohesive pieces of code
- Refactor code following established best practices such as those in *The Art of Readable Code* to improve readability
  - In particular, after making substantial additions or changes to the codebase, refactor the entire codebase in light of those changes

## About Animation

- The physical model of particles in the animation must always match the physical model of particles in the simulation
  - Whenever you resolve a semantic difference between the two physical models to keep them in sync, you must explicitly declare that you have done so

# Supported Command-Line Tools & Usage Guidelines

You have access to the following specialized command-line tools. To minimize token consumption, reduce execution latency, and ensure semantic accuracy, you **must prioritize** these tools over standard Unix commands (like `grep`, `find`, or `git diff`) according to the guidelines below.

- **ripgrep (`rg`)**: Your primary tool for fast text/regex search across the repository.
  - *When to use:* Searching for strings, patterns, or TODOs. Prefer this over `grep -r`. Respects `.gitignore` by default.
  - *Examples:*
    - `rg 'pattern'`
    - `rg -n --glob '*.ts' 'foo'`
    - `rg -l 'TODO'` (list files only)
    - `rg -F 'literal string'` (disables regex)

- **fd (`fdfind`)**: Your primary tool for locating files or directories.
  - *When to use:* Finding specific files by name or extension. Prefer this over `find`. Respects `.gitignore` and uses smart case-matching.
  - *Examples:*
    - `fdfind config`
    - `fdfind -e py` (by extension)
    - `fdfind -t d src` (directories only)
    - `fdfind -H` (includes hidden files)

- **ax**: A local HTTP and HTML I/O utility optimized for AI agents.
  - *When to use:* Performing local web/API requests. Use this instead of writing throwaway curl commands or Python scripts. Run `ax agent-context` first to learn its capabilities.

- **ast-grep (`sg`)**: AST-based structural code search and rewriting.
  - *When to use:* When regular expression search is too fragile (e.g., finding syntax patterns regardless of whitespace or formatting).
  - *Examples:*
    - `sg -p 'console.log($ARG)' -l ts` (structural search)
    - `sg -p 'foo($A)' -r 'bar($A)' -U` (in-place AST rewrite)

- **sem**: Entity-level semantic version control.
  - *When to use:* Analyzing impact, diffs, or git history at the code-entity level (functions, classes, structs) rather than raw lines. Prefer this over `git diff` or `git blame` when structural impact matters.
  - *Examples:*
    - `sem diff` / `sem diff --staged`
    - `sem impact [entity_name]` (simulate impact and dependencies)
    - `sem context [entity_name] --budget 4000` (retrieve token-budgeted context)