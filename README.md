# Laya fine-tuned for Work IQ routing

We fine-tuned [Laya](https://github.com/NandhaKishorM/laya) (`convaiinnovations/laya`, 421M parameters) to route Microsoft 365 workplace questions to the right Work IQ source. The training data is in `data/`.

## Data

| File | Rows | Questions |
|---|---|---|
| `data/train.jsonl` | 2,090 | 16,320 |
| `data/val.jsonl` | 232 | 1,856 |

Each row is one routing request in Laya's training format:

```json
{"state": {...}, "questions": {"workplace_path": {...}, ...}, "gold": {"workplace_path": {"probabilities": {"me": 0.93, ...}}}}
```

How we built it:

1. Generated 1,175 workplace questions across 26 topics (profile, calendar, mail, Teams, files, people, multi-part asks and so on) with `gpt-5.4-mini`. All names are fictional (Contoso, Fabrikam, Megan Bowen and so on).
2. Dropped the 77 questions too close to one of our 134 test questions, leaving 1,098. Nothing in the test set or near it was trained on.
3. Ran each question through our router to get its routing requests, and labelled every request with answer probabilities from Jev 1.13.0. The labels are soft: the full probability spread, not only the top answer.
4. Split 90/10 by question, so every request for a question sits on one side of the split.

## Training

Full fine-tune with Laya's own trainer and its default recipe (RLCD loss, AdamW, effective batch 64), 3 epochs:

```bash
pip install "laya==0.4.0"
python scripts/finetune.py convaiinnovations/laya laya-routing 3
```

On one A100 this took 17 minutes. I ran it on an Azure Container Apps serverless GPU: one container trains, then serves the checkpoint, with no VM or driver setup and billing per second. It also runs on Apple Silicon (MPS), but much more slowly.

## Results

Validation on the 1,856 held-out questions:

| | Before | After |
|---|---|---|
| Accuracy | 0.37 | 0.93 |
| Choice questions | 0.16 | 0.90 |
| Yes/no questions | 0.62 | 0.98 |
| Calibration error (ECE) | 0.33 | 0.09 |

Routing on our 134 test questions, which the model never saw:

| Model | Routed as expected |
|---|---|
| Laya, not fine-tuned | 53 / 134 |
| Laya, fine-tuned | 110 / 134 |
| Jev 1.13.0 | 131 / 134 |

The fine-tuned model handles single questions well across all topics. Its main gap is questions that combine several requests in one sentence ("Who is my manager, and what are my tasks?"): it got 0 of 7.

Speed: about 33 ms per routing request on an A100 and about 150 ms on an M-series Mac GPU.

## Serving

`scripts/serve.py` serves a checkpoint with Jev's request and response format, so a Jev client can point at it:

```bash
API_KEY=change-me python scripts/serve.py laya-routing
curl -X POST localhost:8000/v1/systemone -H "Authorization: Bearer change-me" -d '{"state": {...}, "questions": {...}}'
```

The same script runs in a container. On Azure Container Apps with a GPU workload profile, the app gets an HTTPS endpoint, and swapping in another checkpoint is a new revision.
