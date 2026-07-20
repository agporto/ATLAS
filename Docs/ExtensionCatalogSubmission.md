# Publishing CoherentAtlas in the 3D Slicer Extensions Catalog

CoherentAtlas is prepared for an initial Tier 1 submission to the Slicer ExtensionsIndex.

## Repository configuration

Before updating the ExtensionsIndex pull request:

- Keep the GitHub topic `3d-slicer-extension` on the repository.
- Keep SlicerMorph documented as a complementary extension, not a build dependency.
- Confirm that the raw icon and screenshot URLs in the top-level `CMakeLists.txt` resolve from the `main` branch after merge.
- Rename the repository to `SlicerCoherentAtlas` after the rebrand branch is merged and validated.
- Update all homepage, clone, icon, screenshot, and catalog URLs after the repository rename.

## Catalog entry

Copy `ExtensionsIndex/CoherentAtlas.json` into the root of a fork of `Slicer/ExtensionsIndex`, replacing the obsolete `ATLAS.json` entry in the existing pull request.

The entry uses:

- Extension name: `CoherentAtlas`
- Category: `Registration`
- Revision: `main`
- Build dependencies: none
- Tier: 1

The `scm_url` currently points to `SlicerATLAS.git` while the branch is under review. Change it to `https://github.com/agporto/SlicerCoherentAtlas.git` immediately after the repository rename.

## Minimal validation

Before submission, perform a clean package/install smoke test in a current Slicer Preview build:

1. Configure and package the extension through Slicer's extension build workflow.
2. Install the generated package into a clean Slicer profile.
3. Restart Slicer and open **Atlas Construction**, **Atlas Library**, **Landmark Transfer**, and **Surface Fragmentation**.
4. Confirm that all modules appear under the **CoherentAtlas** category.
5. Confirm that module icons and packaged private Python resources load.
6. Confirm that Landmark Transfer offers installation of `tiny3d` and a compatible `biocpd` release through `slicer.util.pip_install` when they are absent.
7. Load an existing atlas/SSM library created under the earlier ATLAS branding.
8. Run one minimal Construction → Library → Transfer workflow.
9. Run one Surface Fragmentation workflow.

No SlicerMorph dependency or bundled test dataset is required for the initial Tier 1 submission.

## ExtensionsIndex pull-request notes

State that:

- The extension has been renamed from ATLAS to CoherentAtlas before catalog publication.
- Internal Slicer module IDs are collision-safe and extension-specific.
- Existing scientific output filenames and SSM library schemas remain compatible.
- The extension operates locally and does not upload user data.
- Python dependencies are obtained from the configured Python package index with explicit user approval.
- The source is released under the BSD 2-Clause License.
