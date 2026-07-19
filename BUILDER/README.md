# BUILDER lineage and attribution

The ATLAS **BUILDER** module was originally adapted from the Dense Correspondence Landmarking (**DeCAL**) workflow in [SlicerDenseCorrespondenceAnalysis](https://github.com/SlicerMorph/SlicerDenseCorrespondenceAnalysis), developed by the SlicerMorph project.

BUILDER retains the central DeCAL concept of using sparse landmark correspondences to align specimens, deform a template surface, and export index-consistent dense correspondence points. It has since been substantially restructured and extended for the ATLAS workflow.

Major differences and additions include:

- a focused atlas-construction workflow designed to feed the ATLAS DATABASE and PREDICT modules;
- robust, case-insensitive model-landmark pairing across supported file formats;
- coordinate-system validation for detecting likely landmark/mesh orientation mismatches;
- mesh cleaning and warnings for unusually dense surfaces;
- pre-alignment to the specimen closest to the Procrustes mean before mean-atlas construction;
- optional biharmonic deformation with automatic thin-plate spline fallback;
- explicit point-count preview and index-stable dense correspondence export; and
- standardized output directories for aligned models, aligned landmarks, atlas assets, and population correspondences.

Users of BUILDER should cite both ATLAS and the relevant SlicerDenseCorrespondenceAnalysis/DeCA software and publication.

The upstream BSD 2-Clause copyright and license notice is reproduced in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).