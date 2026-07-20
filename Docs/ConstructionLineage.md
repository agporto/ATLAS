# Atlas Construction lineage and attribution

The CoherentAtlas **Atlas Construction** module was originally adapted from the Dense Correspondence Landmarking (**DeCAL**) workflow in [SlicerDenseCorrespondenceAnalysis](https://github.com/SlicerMorph/SlicerDenseCorrespondenceAnalysis), developed by the SlicerMorph project.

Atlas Construction retains the central DeCAL concept of using sparse landmark correspondences to align specimens, deform a template surface, and export index-consistent dense correspondence points. It has since been substantially restructured and extended for the CoherentAtlas workflow.

Major differences and additions include:

- a focused population-atlas workflow designed to feed the CoherentAtlas Library and Landmark Transfer modules;
- robust, case-insensitive model-landmark pairing across supported file formats;
- coordinate-system validation for detecting likely landmark/mesh orientation mismatches;
- mesh cleaning and warnings for unusually dense surfaces;
- pre-alignment to the specimen closest to the Procrustes mean before mean-atlas construction;
- optional biharmonic deformation with automatic thin-plate-spline fallback;
- explicit point-count preview and index-stable dense correspondence export; and
- standardized output directories for aligned models, aligned landmarks, atlas assets, and population correspondences.

Users of Atlas Construction should cite both CoherentAtlas and the relevant SlicerDenseCorrespondenceAnalysis/DeCA software and publication.

The upstream BSD 2-Clause copyright and license notice is reproduced in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
