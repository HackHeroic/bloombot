# BloomBot on a Hugging Face Docker Space (free CPU): the API and the web UI on port 7860.
# Built from a staging folder made by scripts/deploy_space.py (code, trained model, UI build).
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    # public server: no LLM on 2 shared CPUs, papers and profiles kept in memory only
    BLOOMBOT_LLM_POLICY=never BLOOMBOT_STORE=0 BLOOMBOT_WARMUP=1

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only torch keeps the image small
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install -e ".[neural]" && python -m spacy download en_core_web_sm

USER user
# the sentence encoder EmbedLex uses, fetched at build time instead of on the first request
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')"
COPY --chown=user . .

EXPOSE 7860
CMD ["bloombot", "serve", "--host", "0.0.0.0", "--port", "7860"]
