<p align="center">
  <img src="logo.png" alt="CoherentAtlas logo" width="220">
</p>

<h1 align="center">CoherentAtlas</h1>

<p align="center"><strong>Population-informed atlas construction, landmark transfer, and surface fragmentation for 3D biological form</strong></p>

<p align="center">
  A 3D Slicer extension for building population atlases, managing reusable statistical shape model libraries, registering new specimens, transferring landmarks, and generating corresponding surface fragments.
</p>

<p align="center">
  <a href="tutorial/README.md"><strong>Tutorial</strong></a> ·
  <a href="https://github.com/agporto/SlicerATLAS/issues"><strong>Issues</strong></a> ·
  <a href="https://discourse.slicer.org/"><strong>Slicer Forum</strong></a>
</p>

<p align="center">
  <img src="tutorial/images/20.png" alt="CoherentAtlas landmark-transfer result in 3D Slicer" width="760">
</p>

## Overview

CoherentAtlas provides an integrated workflow for population-based 3D morphometrics. It supports:

- construction of mean atlas surfaces and index-consistent dense correspondences from meshes and sparse landmarks;
- creation, storage, loading, and exploration of reusable PCA statistical shape model libraries;
- single-specimen and batch landmark transfer using rigid and shape-model-guided deformable registration;
- target-specific template optimization, including an optional pose-marginalized expectation-maximization initializer;
- generation of population-consistent surface fragments from dense correspondence trajectories and geometric features; and
- a future path toward model-informed completion of partial 3D shapes.

CoherentAtlas runs within [3D Slicer](https://www.slicer.org/). [SlicerMorph](https://slicermorph.github.io/) is complementary but is not required.

## Installation

### 3D Slicer Extension Manager

Once listed in the official Slicer Extensions Catalog:

1. Open 3D Slicer.
2. Select **View > Extension Manager**.
3. Search for **CoherentAtlas** and click **Install**.
4. Restart Slicer when prompted.
5. Find the modules under the **CoherentAtlas** category.

### Developer installation

```bash
git clone https://github.com/agporto/SlicerATLAS.git
```

In Slicer, open **Developer Tools > Extension Wizard**, choose **Select Extension**, and select the cloned repository folder.

## Workflow

1. **Atlas Construction** aligns training meshes and landmarks, creates a mean atlas surface, and exports sparse and dense correspondences.
2. **Atlas Library** packages population correspondences into reusable PCA statistical shape model libraries and provides interactive mode exploration.
3. **Landmark Transfer** adapts the atlas to individual specimens or batches using rigid registration, statistical-shape-model optimization, coherent deformable registration, and optional surface projection.
4. **Surface Fragmentation** uses correspondence-linked geometric features and graph clustering to generate consistent labeled fragments across specimens.
5. **Shape Completion** is planned as a future module for reconstructing missing geometry under learned population-shape constraints.

A complete illustrated walkthrough is available in the [CoherentAtlas tutorial](tutorial/README.md).

## Modules

### Atlas Construction

Constructs a population atlas from folders of surface models and corresponding sparse landmarks. It selects a representative reference, aligns specimens, creates a mean surface, and derives index-consistent dense correspondences.

The implementation was originally adapted from the Dense Correspondence Landmarking (DeCAL) workflow in [SlicerDenseCorrespondenceAnalysis](https://github.com/SlicerMorph/SlicerDenseCorrespondenceAnalysis), developed by the SlicerMorph project. It substantially restructures and extends that workflow with robust model-landmark pairing, coordinate-system checks, mesh-quality safeguards, revised atlas construction, optional biharmonic deformation with thin-plate-spline fallback, and direct integration with the CoherentAtlas Library and Landmark Transfer modules. See [construction lineage](Docs/ConstructionLineage.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

**Primary outputs**

- aligned models and landmarks;
- `atlas_model.ply`;
- `atlas_sparse_landmarks.mrk.json`;
- `atlas_dense_correspondences.mrk.json`; and
- specimen-level population correspondences.

### Atlas Library

Creates and manages reusable statistical shape model libraries from dense population correspondences. It validates point consistency, computes a retained PCA basis, stores the model and associated atlas assets, and provides interactive principal-component visualization in Slicer.

**Primary outputs**

- `manifest.json`;
- `ssm_model.npz`; and
- the associated template surface and sparse/dense markup files.

### Landmark Transfer

Transfers template landmarks to target surfaces in single or batch mode. The registration pipeline combines point-cloud subsampling, FPFH features, RANSAC and ICP rigid alignment, PCA-guided Coherent Point Drift, optional fine deformation, and optional surface projection.

Two target-specific template-optimization backends are available:

- **FPFH + RANSAC**, the established default; and
- **Pose-marginalized EM**, an experimental initializer that evaluates global pose hypotheses while jointly refining SSM shape and similarity pose.

The module installs missing `tiny3d` and `biocpd` dependencies through Slicer's supported `slicer.util.pip_install` interface after explicit user approval.

### Surface Fragmentation

Generates corresponding surface fragments across specimens using dense correspondence trajectories, multiscale geometric features, neighborhood graphs, and spectral clustering. It exports labeled surface models and a label lookup table. “Fragmentation” here means partitioning surfaces into corresponding labeled regions; it does not imply damaged or disconnected input meshes.

## Related 3D Slicer extensions

- **SlicerMorph** provides a broad ecosystem for geometric morphometrics, landmark management, GPA, ALPACA/MALPACA, visualization, and related workflows. CoherentAtlas complements it with population-atlas construction, SSM-constrained registration, and reusable atlas libraries.
- **SlicerDenseCorrespondenceAnalysis / DeCA** provides dense-correspondence and morphometric-analysis workflows. CoherentAtlas Construction is descended from and substantially extends the DeCAL atlas-construction workflow.
- **ALPACA / MALPACA** provide automated landmark transfer through point-cloud registration and are useful alternatives when a statistical population-shape prior is not required.
- General Slicer registration tools provide reusable rigid and deformable registration components, while CoherentAtlas organizes them into a morphometrics-specific population-atlas workflow.

## Compatibility

The rebrand changes Slicer module identifiers but intentionally preserves scientific data conventions and output schemas. Existing files such as `atlas_model.ply`, `atlas_dense_correspondences.mrk.json`, `ssm_model.npz`, and `manifest.json` remain valid and should continue to load.

## Python dependencies

CoherentAtlas uses Slicer-provided Python, VTK, NumPy, and SciPy. Landmark Transfer additionally requires:

- [`tiny3d`](https://pypi.org/project/tiny3d/) for point-cloud processing and rigid registration; and
- [`biocpd>=1.3`](https://pypi.org/project/biocpd/) for shape-model-guided deformable registration and pose initialization.

## Documentation and support

- [CoherentAtlas tutorial](tutorial/README.md)
- [Issue tracker](https://github.com/agporto/SlicerATLAS/issues)
- [3D Slicer documentation](https://slicer.readthedocs.io/)
- [SlicerMorph tutorials](https://github.com/SlicerMorph/Tutorials)
- [Slicer Forum](https://discourse.slicer.org/) — use the *Morphology* or *Extensions* categories

For errors, open **View > Error Log** in Slicer and include the relevant traceback and reproduction steps in the issue report.

## Citation

Until the formal publication is available, cite the software repository:

```text
Porto, A. CoherentAtlas: Population-informed atlas construction and landmark transfer for 3D biological form.
https://github.com/agporto/SlicerATLAS
```

When using Atlas Construction, also cite the relevant SlicerDenseCorrespondenceAnalysis/DeCA software and publication. Cite 3D Slicer, SlicerMorph when used, and the `biocpd` or `tiny3d` methods as appropriate.

## License

CoherentAtlas is distributed under the [BSD 2-Clause License](LICENSE.txt). Portions of Atlas Construction were adapted from SlicerDenseCorrespondenceAnalysis under its BSD 2-Clause License; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Maintainer

**Arthur Porto**  
Florida Museum of Natural History, University of Florida
