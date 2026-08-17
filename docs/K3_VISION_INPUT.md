# K3 native vision input path

The released K3 multimodal stack runs MoonViT + the multimodal projector first, then expands each media placeholder into the projected 7168-dimensional image rows and feeds that merged embedding sequence into the ordinary K3 language model.

This branch adds the same boundary to the resident C worker as `REQMM`. Text `REQ` is unchanged. A `K3MMF1` sidecar carries only projected media rows; text embeddings are still looked up by the exact C model, and each placeholder is expanded to the corresponding image feature length before prefill. Multimodal requests deliberately re-prefill the full merged prompt until a media-aware prefix fingerprint is implemented.

The CI oracle replaces one placeholder with two external rows that are exactly equal to two known token embeddings from the tiny checkpoint. Therefore the mixed-input prompt must generate the exact same token stream as the pure-token prompt; the gate also repeats this with the verified Q4 speculative draft enabled and rejects malformed/mismatched sidecars.
