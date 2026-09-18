# FengXiao (风小)

A small experimental model based on the SP-JEPA architecture.
Character-level text prediction in vector space.

**Status: preflight. Waiting for hardware.**

## Files
- `identity_*.txt` — self-description corpus
- `logic_generator.py` — logic QA generator
- `prepare_corpus.py` — corpus builder
- `sp_jepa.py` — training
- `retrieve.py` — retrieval + calibration
- `l4_decode.py` — retrieval-based decoding
- `L4_DESIGN.md` — L4 design doc

## Not a chatbot
FengXiao does not generate text token-by-token.
It predicts in vector space and retrieves spans.
