# src/dsv41/vendor/ — the dsv41 family's private kernel substrate

The family-isolation law's first instance (the mm3 one is its sibling, src/mm3/vendor/README.md):
vendored Project Maya kernels under `strata::dsv41k`, built into `strata_dsv41_kernels` and the
`strata-dsv41` executable only. Our ds4_iq fork dropped maya's `act/tu/pexp` and grew
`xf16/light_*/hipk_*`; maya's copy evolved independently (it now serves the mm3 family via
src/mm3/vendor/). The two forks must never merge, share a namespace, or share a target.
