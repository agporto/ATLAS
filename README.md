<p align="center">
  <img src="logo.png" alt="ATLAS logo" width="460">
</p>

<h1 align="center">ATLAS</h1>

<p align="center"><strong>Scaling 3D morphometrics with atlas-based automated landmarking</strong></p>

<p align="center">
  A 3D Slicer extension for anatomical atlas construction, statistical shape modeling, deformable registration, and automated landmark transfer.
</p>

<p align="center">
  <a href="tutorial/README.md"><strong>Tutorial</strong></a> ·
  <a href="https://github.com/agporto/SlicerATLAS/issues"><strong>Issues</strong></a> ·
  <a href="https://discourse.slicer.org/"><strong>Slicer Forum</strong></a>
</p>

<p align="center">
  <img src="tutorial/images/20.png" alt="ATLAS landmark-transfer result in 3D Slicer" width="760">
</p>

## Overview

ATLAS provides an integrated workflow for generating and applying 3D anatomical atlases. It supports:

- construction of mean atlas surfaces and dense correspondences from meshes and sparse landmarks;
- creation and interactive exploration of PCA-based statistical shape models;
- single-specimen and batch landmark transfer using rigid and shape-model-guided deformable registration;
- target-specific template optimization before registration; and
- correspondence-guided segmentation of surface models.

ATLAS runs within [3D Slicer](https://www.slicer.org/). [SlicerMorph](https://slicermorph.github.io/) is recommended for complementary morphometric workflows but is not required.

## Installation

### 3D Slicer Extension Manager

Once listed in the official Slicer Extensions Catalog:

1. Open 3D Slicer.
2. Select **View > Extension Manager**.
3. Search for **ATLAS** and click **Install**.
4. Restart Slicer when prompted.
5. Find the modules under the **ATLAS** category.

### Developer installation

```bash
git clone https://github.com/agporto/SlicerATLAS.git
```

In Slicer, open **Developer Tools > Extension Wizard**, choose **Select Extension**, and select the cloned repository folder.

## Workflow

1. **BUILDER** aligns training meshes and landmarks and creates an atlas surface with sparse and dense correspondences.
2. **DATABASE** converts population correspondences into a PCA statistical shape model and loads it for exploration and registration.
3. **PREDICT** transfers landmarks to individual specimens or batches using rigid registration, SSM-guided CPD, optional fine deformation, and surface projection.
4. **SEGMENTATION** uses dense correspondences to divide homologous surface regions consistently across specimens.

A complete illustrated walkthrough is available in the [ATLAS tutorial](tutorial/README.md).

## Modules

### BUILDER

Constructs an anatomical atlas from folders of surface models and corresponding sparse landmarks. BUILDER selects a representative reference, aligns the specimens, generates a mean surface, and derives index-consistent dense correspondences.

BUILDER was originally adapted from the Dense Correspondence Landmarking (DeCAL) workflow in [SlicerDenseCorrespondenceAnalysis](https://github.com/SlicerMorph/SlicerDenseCorrespondenceAnalysis), developed by the SlicerMorph project. It substantially restructures and extends that workflow for ATLAS, including more robust model-landmark pairing, coordinate-system validation, mesh-quality safeguards, revised atlas construction, optional biharmonic deformation with TPS fallback, index-stable dense correspondence export, and direct integration with the DATABASE and PREDICT modules. See [BUILDER attribution and lineage](BUILDER/README.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

**Primary outputs**

- aligned models and landmarks;
- `atlas_model.ply`;
- `atlas_sparse_landmarks.mrk.json`;
- `atlas_dense_correspondences.mrk.json`; and
- specimen-level population correspondences.

### DATABASE

Builds and stores a PCA statistical shape model from dense population correspondences. DATABASE validates point consistency, computes the retained shape basis, saves the model, and provides interactive principal-component visualization in Slicer.

**Primary outputs**

- `manifest.json`;
- `ssm_model.npz`; and
- the associated template model and markup files.

### PREDICT

Transfers template landmarks to target surface models in single or batch mode. The registration pipeline combines point-cloud subsampling, FPFH features, RANSAC and ICP rigid alignment, PCA-guided Coherent Point Drift, optional fine deformation, and optional surface projection.

PREDICT also provides two target-specific template-optimization backends:

- **FPFH + RANSAC**, the default feature-based SSM search; and
- **Pose-marginalized EM**, an experimental initializer that evaluates global pose hypotheses while jointly refining SSM shape and similarity pose.

**Primary outputs**

- predicted landmark `.mrk.json` files;
- warped template models;
- projected landmark refinements; and
- optional batch mesh exports.

### SEGMENTATION

Segments homologous anatomical regions across meshes using dense correspondence trajectories and graph-based clustering. It exports labeled surface models and a label lookup table.

## Python dependencies

ATLAS uses Slicer-provided Python, VTK, NumPy, and SciPy. PREDICT additionally requires:

- [`tiny3d`](https://pypi.org/project/tiny3d/) for point-cloud processing and rigid registration; and
- [`biocpd>=1.3`](https://pypi.org/project/biocpd/) for shape-model-guided deformable registration and pose initialization.

When these packages are missing or incompatible, PREDICT asks for permission before installing or upgrading them through Slicer's Python environment. An internet connection is therefore required the first time those dependencies are installed.

The Pose-marginalized EM backend is optional. The established FPFH + RANSAC template optimizer remains the default.

## Documentation and support

- [ATLAS tutorial](tutorial/README.md)
- [Issue tracker](https://github.com/agporto/SlicerATLAS/issues)
- [3D Slicer documentation](https://slicer.readthedocs.io/)
- [SlicerMorph tutorials](https://github.com/SlicerMorph/Tutorials)
- [Slicer Forum](https://discourse.slicer.org/) — use the *Morphology* or *Extensions* categories

For errors, open **View > Error Log** in Slicer and include the relevant traceback and reproduction steps in the issue report.

## Troubleshooting

| Issue | Likely cause | Suggested action |
|---|---|---|
| Landmark count mismatch in BUILDER | Input landmark files contain inconsistent numbers of points | Standardize landmark counts and regenerate the affected files |
| Subsampling produces no points | Point density is too low or model scales differ substantially | Increase **Point Density** or enable scaling |
| Poor RANSAC alignment | Feature radii or distance threshold are too restrictive | Increase the normal/FPFH radii or RANSAC distance threshold |
| PCA-CPD stops early | The loaded SSM does not match the template correspondence count | Rebuild or reload the matching database |
| Projection overshoots the surface | Projection distance is too large | Reduce the maximum projection factor |
| Batch cancellation is delayed | The current specimen is completing a long registration step | Allow the current step to finish or reduce RANSAC iterations |

## Citation

Until the formal ATLAS publication is available, cite the software repository:

```text
Porto, A. ATLAS: Automated Template-based Landmark Alignment System.
https://github.com/agporto/SlicerATLAS
```

When using BUILDER, also cite the relevant SlicerDenseCorrespondenceAnalysis/DeCA software and publication. Please additionally cite the methods and software used by the relevant workflow, including 3D Slicer, SlicerMorph when used, and the `biocpd` or `tiny3d` documentation as appropriate.

## License

ATLAS is distributed under the [BSD 2-Clause License](LICENSE.txt). Portions of BUILDER were adapted from SlicerDenseCorrespondenceAnalysis under its BSD 2-Clause License; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Maintainer

**Arthur Porto**  
Florida Museum of Natural History, University of Florida