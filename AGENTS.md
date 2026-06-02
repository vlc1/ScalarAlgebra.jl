# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

**ScalarAlgebra** is a Julia package for scalar algebra operations. The package is in early development stages.

## MCP Servers

### julia-mcp (REQUIRED for Julia work)

Always use julia-mcp for executing Julia code. Do NOT use Bash `julia` commands.

julia-mcp provides:
- Persistent REPL session state across multiple code evaluations
- Efficient package management and compilation caching
- Better integration with the development environment
- Access to interactive Julia development

Use `mcp__julia__julia_eval` to run Julia code. This maintains state, avoids
repeated compilation, and is the standard Julia development tool for this project.
