<!-- ATLAS README -->

# ATLAS: Automated Template-based Landmark Alignment System

ATLAS is a companion extension to **SlicerMorph**, focused on high‑fidelity, scalable generation and transfer of 3D anatomical landmarks and dense correspondences using atlas construction, Statistical Shape Models (SSMs), and PCA‑guided deformable registration.

It streamlines:

* Building an atlas (mean model + dense correspondences) from meshes and sparse landmarks
* Constructing and persisting PCA Statistical Shape Models for variation exploration
* Single and batch automated landmark transfer with robust rigid + PCA‑CPD deformable alignment and optional surface projection
* Continuous optimization of the template (pose + shape) before large batch runs

> If you are new to 3D Slicer or SlicerMorph, begin with SlicerMorph tutorials first. ATLAS assumes you already understand loading models, markups, and basic scene management.

---
## Table of Contents
1. [Installation](#installation)
2. [How to Cite](#how-to-cite)
3. [Updating ATLAS](#updating-atlas)
4. [Module Descriptions](#module-descriptions)
5. [Dependencies](#dependencies)
	* [Automatically Installed Python Packages](#automatically-installed-python-packages)
	* [Optional / Recommended Slicer Extensions](#optional--recommended-slicer-extensions)
6. [Related / Complementary Extensions](#related--complementary-extensions)
7. [Tutorials & Learning Resources](#tutorials--learning-resources)
8. [Important Websites](#important-websites)
9. [Funding Acknowledgement](#funding-acknowledgement)
10. [License](#license)
11. [About](#about)

---
## Installation

### 1. From 3D Slicer Extension Manager (future distribution)
When ATLAS becomes available through the Extension Manager, install it like any other extension. (Currently under active development; interim manual install below.)

### 2. Manual (Developer) Installation
1. Clone the repository outside your Slicer install:
	```bash
	git clone https://github.com/agporto/ATLAS.git
	```
2. Add the cloned top‑level folder to Slicer: `Edit > Application Settings > Modules > Additional Module Paths` → Add path → Restart.
3. Open one of the ATLAS modules (BUILDER, DATABASE, PREDICT) to confirm load.

### 3. Cloud / Remote Environments
If you use a cloud Slicer image (e.g., MorphoCloud) you can clone the repo into a writable workspace and add the module path as above. Ensure outbound network allowed for Python dependency install (see below).

---
## How to Cite

If you use ATLAS in publications, please cite (placeholder until formal publication):
```
Porto, A. (2025). ATLAS: Automated Template-based Landmark Alignment System. GitHub repository. https://github.com/agporto/ATLAS
```
Also cite **SlicerMorph** (Rolfe et al. 2021, Methods in Ecology and Evolution) and 3D Slicer (Kikinis et al. 2014) where appropriate. If you use the underlying PCA‑CPD method (biocpd) or tiny3d backend, consult their respective citations.

Once a preprint / DOI is available, update this section.

---
## Updating ATLAS

* **Manual clone**: `git pull` in the repository folder, then restart Slicer.
* **Built extension**: Rebuild (`cmake --build`) then reinstall package (or refresh build directory path).
* If parameters or data formats change, re‑ingest databases using the DATABASE module.

---
## Module Descriptions

ATLAS organizes functionality into three scripted modules:

### BUILDER
**Category:** Atlas & Correspondence Generation  
**Purpose:** Align a set of meshes and sparse landmark sets; automatically pick a reference (closest to mean), similarity-align all specimens, generate a mean (atlas) surface, and derive dense correspondences via TPS warp + k‑d tree mapping. Optionally downsamples to produce a sparser, index‑stable subset.  
**Key Outputs:** Timestamped output folder containing aligned meshes (`*_align.ply`), aligned landmarks (`*_align.mrk.json`), `mean/atlasModel.ply`, `mean/atlasLMs.mrk.json`, and sampled dense set (`atlas.mrk.json`).  
**Notable Features:** Robust base selection; flexible file stem resolution; index preservation during downsampling.

### DATABASE
**Category:** Statistical Shape Modeling  
**Purpose:** Build a PCA SSM from a folder of dense correspondences; store as a lightweight on‑disk database with manifest; load into scene for interactive mode exploration.  
**Workflow:** Ingest → validate consistent point counts → SVD (variance threshold) → save mean, modes, eigenvalues.  
**Visualization:** Single-PC slider + spinbox; deforms mesh in real time using k‑NN interpolation (weights from inverse distance) between landmark displacements and all vertices.  
**Outputs:** `manifest.json`, `ssm_model.npz`, copied template and markup files.

### PREDICT
**Category:** Automated Landmark Transfer  
**Purpose:** Transfer template sparse landmarks to a target mesh (single or batch) via multi-stage alignment: subsampling & FPFH features → RANSAC + ICP rigid alignment → PCA‑guided Coherent Point Drift (CPD) deformable registration → optional surface projection refinement.  
**Modes:**
* Single specimen alignment (tune parameters, inspect intermediate clouds/models)  
* Batch processing (reuse tuned parameters; cancellation + progress)  
* Template optimization (continuous search in SSM space + RANSAC scoring to refine initial template pose/shape)  
**Outputs:** Predicted landmark `.mrk.json` files, warped template models (scene), optionally refined projected landmarks.

#### Core Algorithm Stack
| Stage | Method | Notes |
|-------|--------|-------|
| Subsampling | Voxel grid (voxel from bounding box / point density) | Reduces complexity |
| Features | FPFH descriptors (tiny3d) | For RANSAC correspondence | 
| Rigid init | RANSAC (edge length + distance check) | Iterative fallback if fitness low |
| Refinement | Point-to-plane ICP | Improves rigid transform |
| Deformable | PCA‑guided CPD (biocpd) | Modes rotated into rigid frame; outlier weighting |
| Warp propagation | k‑NN displacement interpolation | Applies to model + sparse landmarks |
| Projection (optional) | Normal-based bidirectional ray casting | Caps max distance fraction |

#### Key Parameters (Advanced Tab)
* `pointDensity` – affects subsampling voxel size.
* `distanceThreshold` / `ICPDistanceThreshold` – rigid alignment tolerances.
* `alpha`, `beta`, `w`, `tolerance`, `max_iterations` – PCA‑CPD control.
* `skipScaling` – disable pre-normalization of template/target size.
* `projectionFactor` – fraction of model length allowed for projection rays.

---
## Dependencies

ATLAS is pure Python + VTK within Slicer, with two external Python packages used at runtime.

### Automatically Installed Python Packages
On first use of PREDICT, a dialog offers to install:
* **`tiny3d`** – lightweight point cloud / registration toolkit (Open3D style API)
* **`biocpd`** – PCA‑guided CPD atlas registration

If installation is denied or blocked (e.g., no network), deformable or feature stages will fail; install manually in Slicer’s Python environment:
```bash
/path/to/Slicer --python-code "import sys, subprocess; [subprocess.check_call([sys.executable,'-m','pip','install',p]) for p in ('tiny3d','biocpd')]"
```

### Optional / Recommended Slicer Extensions
| Extension | Why |
|-----------|-----|
| SegmentEditorExtraEffects | Enhanced segmentation (if you combine segmentation & landmark workflows). |
| SlicerIGT | Additional registration utilities (for validation / comparison). |
| Sandbox / SurfaceMarkups | If experimenting with alternative markup types. |

None are strictly required.

---
## Related / Complementary Extensions
* **SlicerMorph** – Broader morphology workflows (import, GPA, semi-landmarks, ALPACA).
* **Dense Correspondence Analysis (DeCA)** – Downstream analysis of dense correspondences.
* **ALPACA / MALPACA** (in SlicerMorph) – Alternative automated landmarking strategies; compare performance.

---
## Tutorials & Learning Resources
Dedicated ATLAS tutorials are forthcoming. In the meantime:
* Use SlicerMorph tutorials for general data prep & landmark management: https://github.com/SlicerMorph/Tutorials
* Explore parameter effects: run PREDICT Single mode on a representative specimen and inspect each intermediate node.

Planned documents (placeholders):
* Quick Start Atlas Build
* Batch Landmark Transfer Best Practices
* Template Optimization Guide

---
## Important Websites
* 3D Slicer: https://www.slicer.org
* SlicerMorph: https://slicermorph.github.io
* ATLAS repository: https://github.com/agporto/ATLAS
* Slicer Forum (support/discussion): https://discourse.slicer.org (use the `morphology` or `extensions` categories).

---
## Funding Acknowledgement
If ATLAS development is supported by specific grants, list them here. (Placeholder – update with award numbers if applicable.)

---
## License
No explicit license file is currently included. Add a `LICENSE` (e.g., MIT, BSD-3-Clause, or GPL) to clarify reuse. Until then, reuse rights are ambiguous.

---
## About
**Author / Maintainer:** Arthur Porto  
**Scope:** Atlas construction, SSM building, automated & optimized morphometric landmark transfer.  
**Status:** Active development (2025). Feedback and contributions welcome via Issues and Pull Requests.

---
## Quick Start (Condensed)
```text
1. BUILDER: Provide model + landmark folders → Run → get atlas + dense correspondences.
2. DATABASE: Ingest atlas model + dense + sparse + population folder → Build SSM → Load.
3. PREDICT (Single): Tune parameters, run Rigid (RANSAC+ICP) → Deformable (PCA‑CPD) → optional Projection.
4. PREDICT (Template Optimization): Improve template before large batch.
5. PREDICT (Batch): Apply tuned settings to directory of targets → landmark .mrk.json outputs.
```

---
## Troubleshooting
| Issue | Likely Cause | Remedy |
|-------|--------------|--------|
| Landmark count mismatch (BUILDER) | Inconsistent input landmark files | Standardize counts & regenerate outliers |
| Subsampling produced 0 points | Point density too low / extreme scale difference | Increase pointDensity or disable skipScaling |
| RANSAC fitness low | distanceThreshold too small / poor normals | Increase threshold / radii; check mesh quality |
| PCA‑CPD aborts early | Wrong SSM (point count mismatch) | Rebuild or reload matching database |
| Projection overshoot | projectionFactor too large | Decrease factor (1–3%) |
| Batch cancel slow | Large per-specimen transforms | Allow current specimen to finish; reduce RANSAC iters |

Enable `View > Error Log` in Slicer for stack traces; include logs in Issues.

---
## Roadmap (Planned Enhancements)
* GPU acceleration (TPS, k‑NN) where available
* Parameter preset export/import (JSON)
* CLI batch wrapper (headless)
* Expanded test coverage & CI

---
*Last updated:* 2025-08-23

