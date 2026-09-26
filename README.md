# Context-Aware Video Segmentation & Intelligent Ad Placement

Local starter repository for hoichoi AI Builders Hackathon 2026 — Problem 1.

## Goal
Build an end-to-end, explainable pipeline:

`video -> scene/boundary candidates -> break safety -> pacing constraints -> contextual brand matching -> VMAP + debug JSON -> playable demo`

## Engineering principles
- AI for perception/semantic understanding; deterministic logic for hard constraints.
- Never hard-code timestamps or brand assignments.
- Negative-context conflicts are hard exclusions.
- Brand catalogue is data-driven so unseen brands require zero code changes.
- Every selected break should be explainable and testable.
- Original hackathon media stays outside Git.

## Local assets
Create a sibling directory named `hoichoi-assets` and place supplied MP4s there. Do not commit them.

## Status
Architecture skeleton only. Model/tool choices will be made after inspecting the supplied assets.
