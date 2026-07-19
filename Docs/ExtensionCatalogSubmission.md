# Publishing ATLAS in the 3D Slicer Extensions Catalog

ATLAS is prepared for an initial Tier 1 submission to the Slicer ExtensionsIndex.

## Repository configuration

Before opening the ExtensionsIndex pull request:

- Add the GitHub topic `3d-slicer-extension` to the SlicerATLAS repository.
- Keep `SlicerMorph` documented as a complementary extension, not a build dependency.
- Confirm that the raw icon and screenshot URLs in the top-level `CMakeLists.txt` resolve from the `main` branch.
- Merge this publication-preparation branch into `main`.

## Catalog entry

Copy `ExtensionsIndex/ATLAS.json` into the root of a fork of `Slicer/ExtensionsIndex`, then open a pull request against its `main` branch.

The entry intentionally uses:

- Repository: `https://github.com/agporto/SlicerATLAS.git`
- Category: `Registration`
- Revision: `main`
- Build dependencies: none
- Tier: 1

## Minimal validation

ATLAS retains its existing lightweight scripted-module tests. Before submission, perform a clean package/install smoke test in a current Slicer Preview build:

1. Configure and package the extension through Slicer's extension build workflow.
2. Install the generated package into a clean Slicer profile.
3. Restart Slicer and open BUILDER, DATABASE, PREDICT, and SEGMENTATION.
4. Confirm that module icons and packaged Python resources load.
5. Confirm that PREDICT offers installation of `tiny3d` and a compatible `biocpd` release when they are absent.

No SlicerMorph dependency or bundled test dataset is required for the initial Tier 1 submission.

## ExtensionsIndex pull-request notes

State that:

- The repository follows the recommended `Slicer+ExtensionName` naming convention while the extension remains displayed as ATLAS in Slicer.
- The extension operates locally and does not upload user data.
- Python dependencies are obtained from the configured Python package index with explicit user approval.
- The source is released under the BSD 2-Clause License.
