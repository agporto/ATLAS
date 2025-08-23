# TAPIR: Template Alignment, Projection & Iterative Registration

TAPIR is a novel tool for anatomical landmark alignment and transfer between 3D models. It introduces a unique strategy that combines robust initial matching with advanced, data-driven refinement. TAPIR is the first open-source implementation to feature PCA-based Coherent Point Drift (CPD) registration, allowing the alignment process to be guided by real variation in your dataset. It also includes an innovative template optimization workflow, making it possible to automatically tune and select the best template for your study. TAPIR is designed for both single specimens and large-scale batch processing, making it ideal for modern morphometric research.

## What Makes TAPIR Unique?

1. **PCA-based CPD Registration:** TAPIR is the first open-source tool to implement Principal Component Analysis (PCA) guided Coherent Point Drift, allowing landmark alignment to be informed by the true variation in your dataset. This results in more biologically meaningful and accurate registrations.
2. **Automated Template Optimization:** TAPIR introduces a novel continuous optimization strategy for template selection. Instead of grid search, it uses advanced algorithms (Powell's method or Differential Evolution) to explore the space of shape variation and find the best template, guided by a RANSAC-based matching score. This approach is flexible, efficient, and more powerful than traditional discrete methods.
3. **Stepwise Strategy:** The workflow starts with robust initial matching, refines alignment using learned variation, and ensures final accuracy by projecting landmarks onto the target surface.
4. **Scalable Processing:** Use TAPIR for individual specimens or automate landmark transfer for hundreds of models in a batch.

## Getting Started

TAPIR is a Python module for 3D Slicer. To use it, simply load your 3D models and landmarks in Slicer, open the TAPIR module, and follow the workflow tabs. You can process one model at a time or use the batch tools for larger projects. For more details and step-by-step instructions, see the [online documentation](https://github.com/SlicerMorph/SlicerMorph/tree/master/Docs/TAPIR).

## Credits & License

TAPIR was developed by Arthur Porto. If you use TAPIR in your research, please cite this repository and share your feedback. Contributions, bug reports, and suggestions are welcome! Please add your preferred license (e.g., MIT, GPL) in a `LICENSE` file.
