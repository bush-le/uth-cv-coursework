# SETUP.md — Quickstart & Environment Setup Guide

- **Motivation/Background**: Rapid and reliable environment bootstrapping ensures all developers and AI agents can execute tests, linters, and training pipelines immediately without configuration mismatch.
- **Purpose**: Provide instructions for initializing virtual environments, installing dependencies, configuring pre-commit, and running verification smoke tests.
- **Overview Pipeline**: Clone repo -> Setup Python 3.10+ virtualenv -> Install requirements tiers -> Verify via `make test-smoke`.
- **Detailed Plan**: §1 Prerequisites; §2 Quickstart with Makefile; §3 Manual Setup; §4 Verification Steps; §5 Common Troubleshooting.
- **References**: `Makefile`, `pyproject.toml`, `requirements.txt`, `.agents/TROUBLESHOOTING.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-02T07:22:00+07:00

---

## 1. Prerequisites

- Linux (x86_64 or aarch64) or macOS / Windows WSL2.
- Python 3.10, 3.11, or 3.12.
- CUDA 11.8+ / 12.1+ (optional, for GPU acceleration).
- Git & Make.

---

## 2. Quickstart with Makefile

```bash
# 1. Khởi tạo virtual environment và cài đặt dependencies
make install
make install-dev

# 2. Cài đặt pre-commit hooks
make setup-hooks

# 3. Chạy smoke tests kiểm tra môi trường (< 5s)
make test-smoke
```

---

## 3. Manual Setup

```bash
# 1. Tạo môi trường ảo .venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài đặt thư viện
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements/deeplearning.txt -r requirements/tracking.txt

# 3. Kích hoạt pre-commit
pre-commit install
```

---

## 4. Verification Steps

Sau khi cài đặt, hãy kiểm tra tính toàn vẹn của mã nguồn:

```bash
# Kiểm tra định dạng và linter
make lint

# Chạy toàn bộ unit test suite
make test
```
