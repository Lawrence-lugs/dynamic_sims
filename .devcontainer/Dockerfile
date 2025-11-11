FROM mambaorg/micromamba:latest

WORKDIR /usr/src/app

COPY env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes