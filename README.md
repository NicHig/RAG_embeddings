# Semantic Vector Data Asset

This package builds a plaintiff-scoped semantic vector SQLite asset from two existing source databases:

- `corpus.sqlite`
- `unit_text.sqlite`

It does not provide retrieval or serving. v1 is limited to producing and validating `semantic_index.sqlite`.

Large ingestions are now processed in two phases:

- windows are streamed into SQLite record-by-record
- embeddings are generated from pending windows in bounded chunks using `EMBED_BATCH_SIZE` and `EMBED_TOKEN_BUDGET`

If a build is interrupted, incomplete builds can be resumed from the remaining unembedded windows.

## Quickstart

1. Create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

2. Copy `.env.example` to `.env` and set the three database paths plus `OPENAI_API_KEY`.

3. Initialize the output database:

```bash
semantic-index init-db
```

4. Run a sample plaintiff build:

```bash
semantic-index build-plaintiff --cgid SAMPLE-001
```

5. Validate the most recent build:

```bash
semantic-index validate --build-id latest
```

## Commands

- `semantic-index init-db`
- `semantic-index build-full`
- `semantic-index build-plaintiff --cgid <id>`
- `semantic-index build-incremental`
- `semantic-index resume-build --build-id <id|latest>`
- `semantic-index validate --build-id <id|latest>`

## Smoke Test

The test suite includes synthetic `corpus.sqlite` and `unit_text.sqlite` fixtures. Run:

```bash
pytest
```

## Output

The pipeline writes:

- `semantic_builds`
- `semantic_windows`
- `semantic_vectors`
- `semantic_errors`

to `semantic_index.sqlite`.

## Resume Workflow

If a long-running build fails or is interrupted:

```bash
semantic-index resume-build --build-id latest
semantic-index validate --build-id latest
```

Builds are marked `failed` on embedding errors instead of being left in `running` forever.
