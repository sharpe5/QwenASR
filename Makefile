# QwenASR — Qwen3-ASR pure-Rust inference engine
# Makefile: thin wrappers around cargo for the common workflows.

# Cargo binary name (crates/qwen-asr-cli produces `qwen-asr`).
BIN        = qwen-asr
RELEASE_BIN = target/release/$(BIN)

# Release builds target the native CPU for the hand-tuned NEON/AVX kernels
# (matches the README build guidance). Override e.g. RUSTFLAGS= for portable builds.
RUSTFLAGS ?= -C target-cpu=native
export RUSTFLAGS

# Model dir for `make run`. Auto-detects whichever is present (prefers 1.7B);
# override on the CLI: make run MODEL_DIR=path/to/model INPUT=audio.wav
MODEL_DIR ?= $(firstword $(wildcard qwen3-asr-1.7b qwen3-asr-0.6b))

# Model to fetch with `make download` / `make setup`.
MODEL ?= qwen3-asr-0.6b

# Bare `make` should print the target list, not build the first target.
.DEFAULT_GOAL := help

.PHONY: help all setup build release run test fmt fmt-check clippy check clean bench download install uninstall info

# Default: show available targets
all: help

help:
	@echo "QwenASR — Qwen3-ASR pure-Rust inference — Make targets"
	@echo ""
	@echo "Getting started (fresh checkout, zero to transcript):"
	@echo "  make setup     - Release build + download a model (everything to get running)"
	@echo "  make run INPUT=audio.wav   - Transcribe a file (builds if needed)"
	@echo ""
	@echo "Build:"
	@echo "  make build     - Debug build (cargo build)"
	@echo "  make release   - Optimized build, target-cpu=native (cargo build --release)"
	@echo "  make install   - Install the $(BIN) binary to ~/.cargo/bin (cargo install)"
	@echo "  make uninstall - Remove the installed $(BIN) binary (cargo uninstall)"
	@echo "  make download  - Download a model (MODEL=qwen3-asr-0.6b|qwen3-asr-1.7b)"
	@echo ""
	@echo "Test / quality:"
	@echo "  make test      - Run the test suite (cargo test, all crates)"
	@echo "  make fmt       - Format the code (cargo fmt)"
	@echo "  make fmt-check - Check formatting without writing (cargo fmt --check)"
	@echo "  make clippy    - Lint with clippy (warnings as errors)"
	@echo "  make check     - Type-check without building (cargo check)"
	@echo "  make bench     - Run the WER/speed benchmark suite (bench/run.sh)"
	@echo ""
	@echo "Other:"
	@echo "  make clean     - Remove build artifacts (cargo clean)"
	@echo "  make info      - Show build configuration"
	@echo ""
	@echo "Example: make setup && make run INPUT=audio.wav"

# =============================================================================
# Getting started: one command from a fresh checkout to a runnable setup —
# optimized binary built + a model downloaded. Then: make run INPUT=audio.wav
# =============================================================================
setup: release download
	@echo ""
	@echo "Setup complete. Transcribe with:"
	@echo "  make run INPUT=audio.wav"
	@echo "  ./$(RELEASE_BIN) -d $(MODEL) -i audio.wav"

# =============================================================================
# Build
# =============================================================================
build:
	cargo build

release:
	cargo build --release

# Transcribe a file with the release binary. Requires a model and INPUT.
#   make run MODEL_DIR=qwen3-asr-0.6b INPUT=audio.wav
INPUT ?=
run: release
	@test -n "$(MODEL_DIR)" || { \
	  echo "No model dir found (looked for qwen3-asr-1.7b / qwen3-asr-0.6b)."; \
	  echo "Download one:   make download                     # Qwen3-ASR-0.6B (default)"; \
	  echo "          or:   make download MODEL=qwen3-asr-1.7b  # Qwen3-ASR-1.7B"; \
	  echo "Use existing:   make run MODEL_DIR=/path/to/model INPUT=audio.wav"; \
	  exit 1; }
	@test -n "$(INPUT)" || { echo "Set INPUT: make run INPUT=audio.wav"; exit 1; }
	./$(RELEASE_BIN) -d "$(MODEL_DIR)" -i "$(INPUT)"

# =============================================================================
# Test / quality
# =============================================================================
# Runs the full cargo test suite. Model-dependent regression tests skip
# automatically when no model (qwen3-asr-0.6b) is present, so this passes on a
# fresh checkout; download a model to exercise the quality regression too.
test:
	cargo test --workspace

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

check:
	cargo check --workspace

# WER / speed benchmark harness (downloads sample data on first run; see bench/).
bench:
	cd bench && ./run.sh

# =============================================================================
# Utilities
# =============================================================================
# Download a model into ./$(MODEL)/ using the release binary's downloader.
download: release
	./$(RELEASE_BIN) download $(MODEL)

# Install the CLI to ~/.cargo/bin (on PATH for most cargo setups).
install:
	cargo install --path crates/qwen-asr-cli

uninstall:
	cargo uninstall qwen-asr-cli

clean:
	cargo clean

info:
	@echo "Workspace: $(shell pwd)"
	@echo "Binary:    $(BIN)"
	@echo "Cargo:     $(shell cargo --version 2>/dev/null || echo 'not found')"
	@echo "Rustc:     $(shell rustc --version 2>/dev/null || echo 'not found')"
	@echo "Model dir: $(if $(MODEL_DIR),$(MODEL_DIR),<none — run: make download>)"
