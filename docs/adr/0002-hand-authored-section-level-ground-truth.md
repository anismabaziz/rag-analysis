# Hand-author section-level ground truth, do not derive it from the loader

Each question in the evaluation set is labeled by hand with a source document and a section
heading string, read off the PDF by a person. The loader's heuristic `current_section` field is
deliberately not used to produce labels.

## Considered Options

Deriving labels from the loader's own section detection was rejected. It is cheaper and always
consistent with the code, but it is circular: where the loader files a paragraph under the
wrong heading, the label would move with it, and the measured ceiling for every architecture
would silently become whatever the loader happens to do. That is the opposite of what this
project exists to find out.

Chunk-level labels were rejected as well. Ground truth that is a function of the chunker cannot
be used to evaluate the chunker, which rules out the chunking comparison entirely.

The cost is real: labeling a stratified set at section level is the slowest part of the project.
It is still the cheapest way to make the metrics mean anything.
