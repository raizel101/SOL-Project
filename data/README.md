# External corpus

The completed extraction is currently at `outputs/auro_guide_corpus/`. Its
`local_rag.sqlite` index is required to run SOL. The small adapter source is now
maintained at `src/sol_chat/adapters/local_rag.py`; the original extraction copy
remains untouched.
`config/local.json` resolves the index relative to the config file.

This refactor does not move, rewrite or duplicate the approximately 2.2 GB
index, original source texts, coverage reports or completed corpus archive.
Do not commit them to Git or upload them with the frontend. Use the coverage
report and original permissions when deciding what may be published.

Changing the dataset is a separate extraction/index/evaluation workflow, not a
normal application launch or source-code formatting operation.
