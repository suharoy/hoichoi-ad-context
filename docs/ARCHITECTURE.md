# Architecture — Initial Contract

1. **Ingestion** — probe media and produce canonical metadata.
2. **Segmentation** — detect visual/semantic boundary candidates.
3. **Audio** — derive speech/silence/dialogue timing signals.
4. **Context** — produce structured scene/activity/context representation.
5. **Break scoring** — estimate interruption safety with auditable component signals.
6. **Optimization** — select breaks subject to hard pacing/ad-load constraints.
7. **Brand matching** — hard-filter negative contexts, then rank eligible brands.
8. **Manifest** — emit standards-oriented VMAP plus debug JSON.
9. **API/UI** — expose a live pipeline and playable demonstration.

No concrete model dependency is frozen yet. Choices should follow asset inspection and measured behaviour.
