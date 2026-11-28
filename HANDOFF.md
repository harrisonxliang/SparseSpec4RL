# HANDOFF: build the SparseSpec env, run the gates, send back one file

For Harrison. You do three things on the A100 box: build one conda env, run one command, send back one
tarball. None of this code has run on a GPU yet, so a failure may be a bug in a script rather than in
your setup. When a script stops and prints the command that fixes the problem, run that and start it
again. For any other failure, do not fix anything: send the log or the bundle and stop.

Everything is run from the repo root.

```bash
git fetch origin && git checkout sparsespec-verl-backend && git pull
```

## 1. Build the `verl-sparsespec` env (once, roughly 30 to 60 minutes)

It is a copy of `verl-baseline` plus the FlashInfer fork (`3rdparty/flashinfer`), RAFT headers and
`librmm.so` (`3rdparty/raft`), this repo, the patched verl and pytest. It does **not** contain the vLLM
fork. `verl-baseline` is only read, never changed.

```bash
bash integration/build_sparsespec_env.sh --check     # read-only, a few minutes. Fix what it calls FAIL, rerun.
bash integration/build_sparsespec_env.sh             # the build. If it stops, rerun the same command.
bash integration/setup_env.sh verl-sparsespec        # installs the patched verl 0.5.0 into the new env
bash integration/build_sparsespec_env.sh --verify    # must end with an OK line
```

`--check` needs: conda, the `verl-baseline` env, a CUDA toolkit with `nvcc` (same major version as
torch's CUDA), gcc 9.3 or newer, network access, about 40 GB of free disk. It says what is missing and
how to get it.

The build and `--verify` end with a block "next", whose step 3 is one line starting
`unset INSTALL_PREFIX PREFIX; export CUDA_HOME=...`. **Keep that line.** You paste it before the gate
run and before the real run: the engine compiles its GPU kernels at first start and needs the same
CUDA toolkit as the build.

Then upload the env to the shared Drive: the build ends by printing the exact packaging commands
(conda-pack into `~/verl-sparsespec-pack/`). Run them and upload that folder.

If a step fails: send the last screen and the newest file in `integration/logs/`.

## 2. Before the gate run

| Need | How to check |
|---|---|
| The whole GPU free. No baseline training may be running; the script refuses to start if more than 1.5 GiB is in use, and it never touches anyone else's process. | `nvidia-smi` |
| About 32 GiB of host RAM available (the engine page-locks 16 GiB for its offload pool). | `free -g`, column "available" |
| About 15 GB free where checkpoints go, 5 GB in the repo. | `df -h ~` |
| The baseline's step-25 checkpoint at `$CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25` (default `CKPT_ROOT` is `~/ckpts/sparsespec_baseline`). If it is missing, that one gate is skipped and the rest still runs. | `ls ~/ckpts/sparsespec_baseline/vllm_baseline_seed0/hf/` |
| The training data where the baseline has it (`~/data/sparsespec_baseline`): `math_l3to5_train.parquet` and `math500_test.parquet`. | `ls ~/data/sparsespec_baseline/*.parquet` |

If your checkpoints or data live somewhere else, `export CKPT_ROOT=...` / `export DATA_DIR=...` first,
the same way as for the baseline.

## 3. The one command

```bash
tmux new -s gates                      # it runs for hours and must survive your SSH session
# inside tmux (a new tmux session does not inherit your exports): paste the line from the build's step 3
unset INSTALL_PREFIX PREFIX; export CUDA_HOME=...
bash integration/run_all_gates.sh
```

No conda env needs to be active; the script switches between `verl-sparsespec` and `verl-baseline`
itself. If something you can fix in a minute is missing, it stops in the first minutes and prints the
command to run. Otherwise it keeps going after a failed stage and prints a PASS/FAIL line per stage.

Rough durations (estimates, nothing has been timed):

| Stage | Time |
|---|---|
| Versions, CPU tests, import check | 5 to 10 min |
| Reference-engine plumbing run | 10 to 30 min |
| GPU unit tests (four variants) | 1 to 2 h |
| Gate 3b: stock vLLM, then the three-way comparison | 15 to 40 min |
| Gate 3c: synthetic checkpoint, stock-vLLM reference, step-25 checkpoint | 1 to 2.5 h |
| Smoke runs: 3 steps default, 3 steps host_guard, 3 steps without CUDA graphs, 1 step without log-probs | 3 to 8 h |
| **Total** | **5 to 12 h** |

## 4. What to send back

The last lines of the output are a one-screen summary and

```
send this file back: <repo>/baseline/results/gate_bundles/gates_<host>_<time>.tar.gz
```

Send that file (Drive or chat) and paste the summary. Send it whatever the result, also after a Ctrl-C or
a refusal to start: the bundle is written in every case and holds all logs and tracebacks.

## 5. What NOT to do

- Do not edit `baseline/config/grpo_base.sh`, `baseline/reward.py`, the data prep or the baseline scripts.
- Do not install or change anything in the `verl-baseline` env.
- Do not kill, renice or otherwise touch other people's jobs. If the GPU is busy, wait and rerun.
- Do not change a threshold or skip a check to get a PASS, and do not retry a failed gate by hand.
- **Do not launch the real 100-step run until Dylan confirms the gates passed.**

## 6. The real run (only after Dylan's go-ahead)

```bash
tmux new -s sparsespec
unset INSTALL_PREFIX PREFIX; export CUDA_HOME=...    # the same line as for the gate run
conda activate verl-sparsespec
nvidia-smi                              # the card must be free
bash baseline/run_seed_sparsespec.sh    # seed 0, 100 steps, about a day
```

Dylan will say whether to put `SPARSESPEC_ADMISSION=host_guard` in front of the last line. Progress:
`tail -f baseline/logs/sparsespec_seed0.log`. If it dies, do not restart it; send the log.
