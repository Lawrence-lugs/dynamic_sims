FROM mambaorg/micromamba:latest
WORKDIR /usr/src/app

# Latex for manim
# USER root
# RUN apt update && apt install -y --no-install-recommends \
#     libcairo2-dev \
#     texlive texlive-latex-extra texlive-fonts-extra \
#     texlive-latex-recommended texlive-science \
#     tipa libpango1.0-dev \
#     && apt clean \
#     && rm -rf /var/lib/apt/lists/*

# Dynamic sims env
COPY env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
micromamba clean --all --yes
USER mambauser