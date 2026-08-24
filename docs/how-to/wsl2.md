# Run HFlow on Windows with WSL2

Run the full HFlow development loop inside WSL2 on Windows 11. This is the
only supported Windows path: native Windows fails because HFlow imports
`fcntl` for file locking (`src/hflow/storage.py`), which does not exist on
Windows.

## 1. Install WSL2 and Ubuntu

From PowerShell (admin):

```powershell
wsl --install -d Ubuntu
```

Restart when prompted. On first launch, create your Ubuntu username and
password.

## 2. Install uv inside WSL

From the WSL terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add `~/.local/bin` to your PATH so `uv` is available in new shells:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
uv --version
```

## 3. Install ffmpeg

```bash
sudo apt update && sudo apt install -y ffmpeg
```

Verify:

```bash
which ffmpeg
ffmpeg -version
```

## 4. Clone the repo inside WSL filesystem

Clone inside the Linux filesystem (`~/hflow`) for fast I/O. **Do not clone
under `/mnt/c`** — file metadata, symlinks, and lock performance are far
slower there.

```bash
git clone https://github.com/Hebbian-Robotics/hflow.git ~/hflow
cd ~/hflow
```

If the repo already exists on the Windows side (e.g. `/mnt/c/Users/you/hflow`),
you can work from there but expect slower `uv sync`, `pytest`, and `uv run`
times. If you do, keep your data root (`./data`) on the Linux side:

```bash
# Only if ~/hflow does not already exist
ln -s /mnt/c/Users/you/hflow ~/hflow
```

## 5. Install the locked development environment

```bash
uv sync --locked
```

## 5. Verify the install

```bash
uv run python -c "import hflow; print(hflow.__version__)"
uv run hflow --help
```

## 6. Run the quality gates

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest -q
```

Two integration tests are opt-in (require network or Docker):

```bash
HFLOW_NETWORK_TESTS=1 uv run pytest tests/test_ffmpeg.py -q
HFLOW_DOCKER_TESTS=1 uv run pytest tests/test_runtime_integration.py -q
```

## 6. Run the quickstart

```bash
uv run python examples/quickstart.py
```

Output: `data/sample/episode_0001.mcap` and a canonical episode under
`data/test-runs/episode_0001-<hash>/episode_0001.canonical.mcap`
with measurements printed to the terminal.

## 7. Run hflow doctor on an episode

```bash
uv run hflow doctor data/test-runs/episode_0001-*/episode_0001.canonical.mcap
```

Expected output:

```text
doctor: data/test-runs/episode_0001-*/episode_0001.canonical.mcap
  conforming: no findings
  verdict: CONFORMING
```

## 7. Troubleshooting

### `ffmpeg not found`

- `sudo apt install -y ffmpeg` inside WSL
- Verify `which ffmpeg` prints a path

### `uv: command not found`

- `export PATH="$HOME/.local/bin:$PATH"` or reopen the shell after adding
  to `~/.bashrc`

### Slow `uv sync`, `pytest`, or file I/O

- You are likely running from `/mnt/c` (Windows drive). Move the clone to
  `~/hflow` (Linux filesystem).
- Keep the data root (`./data`) on the Linux filesystem as well.

### `ImportError: cannot import name 'fcntl'`

- You are running from Windows PowerShell or CMD. Use the WSL terminal
  (`wsl` or Windows Terminal > Ubuntu).

### Docker / Airflow tests fail

- Ensure Docker Desktop has WSL2 integration enabled for Ubuntu.
- Run `HFLOW_DOCKER_TESTS=1 uv run pytest tests/test_runtime_integration.py -q`
  from the WSL shell.

---

## Appendix: Windows-side editor with WSL backend

Use VS Code with the "WSL" extension:

1. Install "WSL" extension in VS Code (ms-vscode-remote.remote-wsl).
2. `Ctrl+Shift+P` → "WSL: Connect to WSL"
3. Open `~/hflow` from the WSL side.

The terminal in VS Code will be a WSL shell; `uv`, `pytest`, and `hflow`
commands run natively inside WSL.

---

## Link from docs index

Add to `docs/README.md` under **How-to guides**:

- [Run HFlow on Windows with WSL2](./wsl2.md)