# WS16 Pod Session Plan

This document is the operational plan for completing `WS16` on a RunPod or
equivalent GPU environment. It exists so the session can resume cleanly without
reconstructing the workflow from chat history.

## Primary Objective

Complete `WS16` cleanly, save or push the sign-off state, and only then use any
remaining pod time for structured post-`WS16` work.

## Known Resume State

From the previous pod session, the following were already validated:

- repo/bootstrap/GPU bring-up
- semantic cache for `BAAI/bge-small-en-v1.5`
- full suite on pod: `188 passed, 8 warnings`
- payload reproducibility complete
- runtime sandbox policy correct
- NLP canonical refusal truth correct
- QML canonical execution correct

The only required `WS16` item still open is:

- managed endpoint lifecycle validation

## Session Success Definition

The session is successful only if all of these are true:

- pod environment is usable
- TAR status works
- managed endpoint lifecycle is validated:
  - register
  - start
  - healthy
  - restart
  - healthy again
  - stop
  - persisted endpoint state/logs visible
- `WS16` sign-off note is saved
- repo state is cleanly saved locally and pushed if desired

## Recommended Pod Spec

- GPU: `A40 48GB` or `L40S 48GB`
- storage: `100-150GB` persistent volume minimum
- OS: standard Linux image with Python 3.11 and Docker support
- session budget: `6 hours` safe target

## Phase 0: Pod Bring-Up

Expected time:

- `15-30 min` if persistent volume exists
- `45-90 min` if fresh pod

Goal:

- determine whether the pod is a resume or rebuild session

Run:

```bash
pwd
python3 --version
git --version
nvidia-smi -L
df -h /
```

Then:

```bash
cd /root
test -d Thermodynamic-Continual-Learning-delivered && echo REPO_PRESENT || echo REPO_MISSING
```

Decision:

- if repo exists, resume
- if repo does not exist, clone fresh

## Phase 1: Repo And Environment Restore

If repo exists:

```bash
cd /root/Thermodynamic-Continual-Learning-delivered
git pull
```

If repo is missing:

```bash
cd /root
git clone https://github.com/christophergardner-star/Thermodynamic-Continual-Learning-delivered.git
cd Thermodynamic-Continual-Learning-delivered
```

Check venv:

```bash
test -d .venv && echo VENV_PRESENT || echo VENV_MISSING
```

If `.venv` is missing:

```bash
python3 bootstrap.py --gpu
```

Activate:

```bash
source .venv/bin/activate
```

If the semantic model is not cached yet:

```bash
python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('BAAI/bge-small-en-v1.5'); print('cached', m.get_sentence_embedding_dimension())"
```

## Phase 2: Fast Sanity Check

Expected time:

- `10-20 min`

Run:

```bash
python tar_cli.py --direct --status
python tar_cli.py --direct --runtime-status --json
python -m pytest tests -q
```

Checkpoint:

- if these pass, continue to `WS16` completion
- if tests fail, fix only blockers relevant to `WS16`

## Phase 3: Finish WS16 Endpoint Lifecycle Validation

Expected time:

- `45-90 min`

Create mock endpoint model dir:

```bash
mkdir -p .tmp_ws16/mock-model
printf '{}' > .tmp_ws16/mock-model/mock_endpoint.json
```

Register checkpoint:

```bash
python tar_cli.py --direct --register-checkpoint --checkpoint-name ws16-mock --model-path /root/Thermodynamic-Continual-Learning-delivered/.tmp_ws16/mock-model --backend-name transformers --role-name assistant --json
```

Start endpoint:

```bash
python tar_cli.py --direct --start-endpoint --checkpoint-name ws16-mock --role-name assistant --port 8816 --wait-for-health --json
```

List endpoints:

```bash
python tar_cli.py --direct --list-endpoints --json
```

Check endpoint health:

```bash
python tar_cli.py --direct --endpoint-health --endpoint-name assistant-ws16-mock --json
```

Restart endpoint:

```bash
python tar_cli.py --direct --restart-endpoint --endpoint-name assistant-ws16-mock --wait-for-health --json
```

Check health again:

```bash
python tar_cli.py --direct --endpoint-health --endpoint-name assistant-ws16-mock --json
```

Stop endpoint:

```bash
python tar_cli.py --direct --stop-endpoint --endpoint-name assistant-ws16-mock --json
```

Final list:

```bash
python tar_cli.py --direct --list-endpoints --json
```

## Phase 4: WS16 Sign-Off Check

`WS16` is complete only if all of the following can be stated truthfully:

- pod bootstrap works
- TAR status works
- runtime manifests are reproducibility-complete
- benchmark truth surfaces are honest
- canonical QML path is real and aligned
- managed endpoint lifecycle works on pod
- endpoint health/log/trust policy are observable

If all are true:

- `WS16` is closed

## Phase 5: Save State Properly

Expected time:

- `20-40 min`

Do:

- save or update `docs/ws16_resume_notes.md` into a final sign-off note
- optionally add `docs/ws16_validation_report.md`
- check repo status
- commit only intended docs/code
- push if desired

Core commands:

```bash
git status --short
git add <intended files>
git commit -m "Complete WS16 pod validation"
git push origin main
```

## Phase 6: If Time Remains

Only start one controlled follow-on task.

Best choice:

- define `WS17` execution spec and architecture primitives

Good use of remaining time:

- project object model
- pause/resume semantics
- evidence budget schema
- stop/pivot decision rules

Bad use of remaining time:

- README cleanup
- UI polishing
- random feature additions
- starting multiple new workstreams at once

## Failure Handling

If disk is low:

```bash
df -h /
du -sh /root/.cache 2>/dev/null || true
du -sh .venv 2>/dev/null || true
```

If bootstrap fails from space:

- remove failed `.venv`
- clear pip cache
- rerun bootstrap

If `tar_cli.py --status` fails:

- fix only the immediate blocker
- re-run status before anything else

If endpoint start fails:

- inspect endpoint logs in `tar_state/endpoints/...`
- verify manifest path
- verify health output
- do not move on until the failure reason is understood

## Time Budget

Recommended session allocation:

- Phase 0-2: `1 to 2 hours`
- Phase 3: `1 to 1.5 hours`
- Phase 4-5: `30 to 60 min`
- buffer: `2+ hours`

This is why `6 hours` is the safe session target.

## Hard Stop Rules

Do not start post-`WS16` work unless:

- endpoint lifecycle validation is done
- repo state is saved
- `WS16` sign-off is explicit

## Best Session Outcome

Ideal outcome:

1. reconnect or rebuild pod
2. finish endpoint lifecycle validation
3. close `WS16`
4. save or push the final state
5. use leftover time to design `WS17`, not to sprawl
