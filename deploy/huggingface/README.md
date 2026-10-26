---
title: BloomBot
emoji: 🪜
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Check exam papers by Bloom level, time and marks
---

# BloomBot

Classifies exam questions by revised Bloom level, checks a paper against a blueprint, estimates
whether it fits its exam time, checks its marks, and rewrites questions to rebalance it.

This Space runs the API and the web UI on a free CPU, in **fast mode** (no LLM). Papers you upload
are processed in memory and not stored. For the LLM second opinion and LLM rewrites, run BloomBot
on your own computer: https://github.com/HackHeroic/bloombot

* Web UI: this page, or https://bloombot.muralimadhav.com
* API docs: `/docs` on this Space

The labelled questions shipped here (in the trained model's example bank and the dataset explorer)
come from six public sources, each under its own licence: see [NOTICE.md](NOTICE.md).
