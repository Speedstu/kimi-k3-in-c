# Vision parity status

K3 image input now has an exact language-model boundary in the C runtime. The resident worker accepts projected image rows through `REQMM`; hosted CI proves that replacing one media placeholder with external rows identical to known token embeddings produces the exact same generated token stream, including with verified speculative decoding enabled.

The local bridge also contains a selective official K3 frontend: it lazy-loads the released `MoonViT3dPretrainedModel` and `PatchMergerMLPV2` weights plus the official processor, without instantiating a second 2.78T PyTorch language model. The projected 7168-dimensional rows are handed to the exact C language model.

This is **not yet a claim that the published vision benchmark scores have been reproduced**. Loading the actual full-checkpoint MoonViT/projector and measuring the official vision suites require the self-hosted full-checkpoint runner. Those gates remain fail-closed. Video/audio are also rejected until their released preprocessing path is implemented and parity-gated.
