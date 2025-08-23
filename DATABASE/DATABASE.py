import os
import numpy as np
import shutil
import logging
import json
from pathlib import Path
from vtk.util import numpy_support

from scipy.spatial import cKDTree

import slicer, qt, ctk, vtk
from slicer.ScriptedLoadableModule import *

#
# DATABASE
#

class DATABASE(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DATABASE" # Changed Title
    self.parent.categories = ["ATLAS"]
    self.parent.dependencies = []
    self.parent.contributors = ["Arthur Porto"] # Acknowledged contribution
    self.parent.helpText = """
    This module builds a Statistical Shape Model (SSM) from a population of landmark files
    and saves it to a persistent local database. It also provides a visualization tool
    to explore the modes of variation of a loaded SSM.
    """
    self.parent.acknowledgementText = ""

#
# DATABASELogic
#

class DATABASELogic(ScriptedLoadableModuleLogic):
  def __init__(self):
      self.is_cancelled = False

  @staticmethod
  def build_ssm(shapes, variance_threshold=0.99, max_modes=None):
    """
    Builds a statistical shape model using Principal Component Analysis (PCA).

    Args:
        shapes (np.ndarray): A numpy array of shape (n_shapes, n_points, 3)
                            containing the landmark coordinates for the population.
        variance_threshold (float, optional): The cumulative variance to be explained
                                              by the selected modes. Defaults to 0.99.
        max_modes (int, optional): The maximum number of modes to retain, regardless
                                   of variance. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The mean shape (n_points, 3).
            - np.ndarray: The modes of variation (eigenvectors) (n_points, 3, n_modes).
            - np.ndarray: The eigenvalues for the selected modes.
            - int: The number of modes retained in the model.
    """
    n_shapes, n_pts, dim = shapes.shape
    X = shapes.reshape(n_shapes, -1).astype(np.float64, copy=False)
    mu = X.mean(axis=0); Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    if np.allclose(s, 0):
        mean = mu.reshape(n_pts, dim).astype(np.float32, copy=False)
        modes = np.zeros((n_pts, dim, 0), np.float32)
        eig = np.zeros((0,), np.float64)
        logging.debug("SSM: all shapes identical; no modes.")
        return mean, modes, eig, 0
    eigvals = (s**2) / max(n_shapes-1, 1)
    var_ratio = eigvals / eigvals.sum()
    if max_modes is not None: num_modes = min(max_modes, len(eigvals))
    elif variance_threshold is None or variance_threshold >= 1.0: num_modes = len(eigvals)
    else: num_modes = np.searchsorted(np.cumsum(var_ratio), variance_threshold) + 1
    scales = np.maximum(s[:num_modes], np.finfo(float).eps)
    modes_flat = (Xc.T @ U[:, :num_modes]) / scales
    modes = modes_flat.reshape(n_pts, dim, num_modes).astype(np.float32, copy=False)
    mean_shape = mu.reshape(n_pts, dim).astype(np.float32, copy=False)
    logging.debug(f"SSM modes={num_modes} explained={var_ratio[:num_modes].sum()*100:.1f}%")
    return mean_shape, modes, eigvals[:num_modes].astype(np.float64, copy=False), num_modes

  def _norm_cs(self, cs):
    s=str(cs).upper()
    if 'LPS' in s or s=='0': return 'LPS'
    if 'RAS' in s or s=='1': return 'RAS'
    return 'RAS'
  
  def _read_fcsv(self, path):
    cs='RAS'; pts=[]
    with open(path,'r') as f:
        for line in f:
            if line.startswith('#'):
                if 'CoordinateSystem' in line:
                    cs=self._norm_cs(line.split('=')[-1])
                continue
            parts=line.strip().split(',')
            if len(parts)>=4:
                try: pts.append([float(parts[1]),float(parts[2]),float(parts[3])])
                except (ValueError, IndexError) as e:
                    logging.warning(f"Could not parse point in file {path} on line: '{line.strip()}'. Error: {e}")
    return np.asarray(pts, dtype=float), cs
  
  def _read_json(self, path):
    data=json.load(open(path,'r'))
    m=(data.get('markups') or [{}])[0]
    cs=self._norm_cs(m.get('coordinateSystem','RAS'))
    cps=m.get('controlPoints') or []
    pts=np.asarray([cp['position'] for cp in cps], dtype=float)
    return pts, cs

  @staticmethod
  def _to_ras(pts, cs):
    if cs=='LPS':
        a=pts.copy(); a[:,:2]*=-1; return a
    return pts
  def _markups_points_ras(self, filepath):
    if filepath.lower().endswith('.fcsv'):
        pts, cs = self._read_fcsv(filepath)
    else:
        pts, cs = self._read_json(filepath)
    return self._to_ras(pts, cs)

  def _markups_count(self, filepath): return int(self._markups_points_ras(filepath).shape[0])

  def ingestSSMDatabase(self, dbPath, templateModelPath, templateLandmarksPath, sparseLandmarksPath, populationDir, progress_callback=None):
    def update_progress(p, msg):
        if progress_callback:
            progress_callback(p, msg)

    update_progress(0, "Preparing...")
    os.makedirs(dbPath, exist_ok=True)

    try:
        update_progress(5, "Scanning landmark files...")
        landmark_files = sorted(f for f in os.listdir(populationDir) if f.lower().endswith(('.fcsv','.json')))

        if not landmark_files:
            return False, "No markups files found."
        total_files = len(landmark_files)

        all_shapes, point_count = [], -1
        for i, fname in enumerate(landmark_files):
            update_progress(1 + int((i / total_files) * 55), f"Loading: {os.path.basename(fname)}")
            path = os.path.join(populationDir, fname)

            # Load and validate shape
            current_points = self._markups_points_ras(path)

            if current_points.ndim != 2 or current_points.shape[1] != 3:
                return False, f"Invalid shape in {fname}: got {current_points.shape}"
            if not current_points.any():
                continue
            if point_count == -1:
                point_count = len(current_points)
            elif len(current_points) != point_count:
                return False, f"Inconsistent landmark count in {fname}."
            all_shapes.append(current_points.reshape(-1, 3))

        if not all_shapes:
            return False, "No valid landmark files were loaded."
        shapes_np = np.stack(all_shapes, axis=0)  # (n_shapes, n_points, 3)

    except Exception as e:
        return False, f"Error loading landmarks: {e}"

    try:
        update_progress(55, "Building Statistical Shape Model...")
        mean_shape, modes, eigenvalues, num_modes = self.build_ssm(shapes_np)
        update_progress(85, f"SSM built with {num_modes} modes.")
    except Exception as e:
        return False, f"Error building SSM: {e}"

    try:
        update_progress(90, "Saving database files...")
        mext=self._suffixes(templateModelPath)
        dext=self._suffixes(templateLandmarksPath)
        sext=self._suffixes(sparseLandmarksPath)
        modelFile=f"template_model{mext}"
        denseFile=f"dense_correspondences{dext}"
        sparseFile=f"sparse_landmarks{sext}"
        shutil.copy2(templateModelPath, os.path.join(dbPath, modelFile))
        shutil.copy2(templateLandmarksPath, os.path.join(dbPath, denseFile))
        shutil.copy2(sparseLandmarksPath,   os.path.join(dbPath, sparseFile))
        ssm_output_path=os.path.join(dbPath,"ssm_model.npz")
        np.savez_compressed(ssm_output_path,
            mean_shape=mean_shape.astype(np.float32,copy=False),
            modes=modes.astype(np.float32,copy=False),
            eigenvalues=eigenvalues.astype(np.float64,copy=False))
        manifest={
            "version":1,
            "files":{"model":modelFile,"dense":denseFile,"sparse":sparseFile,"ssm":"ssm_model.npz"},
            "counts":{"dense":self._markups_count(os.path.join(dbPath,denseFile)),
                      "sparse":self._markups_count(os.path.join(dbPath,sparseFile))}
        }
        with open(os.path.join(dbPath,"manifest.json"),"w") as f: json.dump(manifest,f, indent=2)
        update_progress(100, "Finalizing...")
    except Exception as e:
        return False, f"Error saving database files: {e}"
    dbName = Path(dbPath).name
    return True, f"Database '{dbName}' saved."


  def removeSSMDatabase(self, dbPath):
    try: shutil.rmtree(dbPath); return True, f"Database removed."
    except Exception as e: return False, f"Error removing database: {e}"


  def loadSSMDatabase(self, dbPath):
      try:
          dbName=Path(dbPath).name
          manifest_path=os.path.join(dbPath,"manifest.json")
          modelFile=lmFile=sparseFile=ssmFile=None
          if os.path.exists(manifest_path):
              man=json.load(open(manifest_path,"r"))
              files=man.get("files",{})
              modelFile=os.path.join(dbPath,files.get("model",""))
              lmFile=os.path.join(dbPath,files.get("dense",""))
              sparseFile=os.path.join(dbPath,files.get("sparse",""))
              ssmFile=os.path.join(dbPath,files.get("ssm","ssm_model.npz"))
          else:
              files=sorted(os.listdir(dbPath))
              models=[f for f in files if f.lower().endswith((".ply",".vtk",".vtp",".stl"))]
              npzs=[f for f in files if f.lower().endswith(".npz")]
              markups=[f for f in files if f.lower().endswith((".fcsv",".json"))]
              modelFile=os.path.join(dbPath,models[0]) if models else None
              ssmFile=os.path.join(dbPath,"ssm_model.npz") if "ssm_model.npz" in npzs else (os.path.join(dbPath,npzs[0]) if npzs else None)
              if len(markups)==1:
                  lmFile=os.path.join(dbPath,markups[0]); sparseFile=None
              elif len(markups)>=2:
                  dense_hint=[f for f in markups if ("dense" in f.lower()) or ("corr" in f.lower())]
                  sparse_hint=[f for f in markups if "sparse" in f.lower()]
                  if dense_hint and sparse_hint:
                      lmFile=os.path.join(dbPath,dense_hint[0]); sparseFile=os.path.join(dbPath,sparse_hint[0])
                  else:
                      counts=[(f,self._markups_count(os.path.join(dbPath,f))) for f in markups]
                      counts.sort(key=lambda x:x[1],reverse=True)
                      lmFile=os.path.join(dbPath,counts[0][0])
                      sparseFile=os.path.join(dbPath,counts[1][0]) if len(counts)>1 else None

          if not (modelFile and ssmFile and lmFile):
            return False, "Database is incomplete."

          ssm_data = np.load(ssmFile, allow_pickle=False)
          mean_shape=ssm_data["mean_shape"]; modes=ssm_data["modes"]; eigenvalues=ssm_data["eigenvalues"]
          needed_pts=int(mean_shape.shape[0])

          if sparseFile:
              n_lm=self._markups_count(lmFile)
              n_sp=self._markups_count(sparseFile)
              if n_lm!=needed_pts and n_sp==needed_pts:
                  lmFile, sparseFile = sparseFile, lmFile

          modelNode=slicer.util.loadModel(modelFile)
          lmNode=slicer.util.loadMarkups(lmFile)
          sparseNode=slicer.util.loadMarkups(sparseFile) if sparseFile else None

          base_name=f"{dbName}_template"
          modelNode.SetName(base_name)
          lmNode.SetName(f"{base_name}_correspondences")
          if sparseNode: sparseNode.SetName(f"{base_name}_sparse_landmarks")

          mean_flat=mean_shape.reshape(-1,1)
          n_modes=modes.shape[2]
          modes_flat=modes.reshape(-1,n_modes)
          table_array=np.hstack([mean_flat,modes_flat])
          tableNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", f"ssm_data_{dbName}")
          slicer.util.updateTableFromArray(tableNode, table_array.astype(np.float32,copy=False))
          t=tableNode.GetTable(); t.GetColumn(0).SetName("mean_shape")
          for i in range(n_modes): t.GetColumn(i+1).SetName(f"mode_{i}")
          tableNode.SetAttribute("ssm_eigenvalues", json.dumps(eigenvalues.tolist()))
          tableNode.SetAttribute("ssm_npoints", str(needed_pts))
          return True, f"SSM '{dbName}' loaded. Table Node has {n_modes} modes."
      except Exception as e:
          return False, f"Error loading database: {e}"
      
  def _suffixes(self, p): 
      s=Path(p).suffixes
      return "".join(s) if s else Path(p).suffix


#
# DATABASEWidget
#

class DATABASEWidget(ScriptedLoadableModuleWidget):
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.logic = DATABASELogic()
    self.ssm_data = {} # To hold loaded SSM data for visualization
    self.pcSelectorSpinBox = None
    self.pcSlider = None

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # --- UI Setup ---
    # (Database Management and Ingestion UI sections remain the same)
    dbWidget = ctk.ctkCollapsibleButton()
    dbWidget.text = "Shape Modeling Database"
    dbLayout = qt.QFormLayout(dbWidget)
    self.layout.addWidget(dbWidget)
    self.databasePathSelector = ctk.ctkPathLineEdit()
    self.databasePathSelector.filters = ctk.ctkPathLineEdit.Dirs
    dbLayout.addRow("Database Location:", self.databasePathSelector)
    self.databaseListWidget = qt.QListWidget()
    self.loadDbButton = qt.QPushButton("Load Selected")
    self.loadDbButton.setIcon(slicer.app.style().standardIcon(qt.QStyle.SP_ArrowUp))
    self.loadDbButton.enabled = False
    self.removeDbButton = qt.QPushButton("Remove Selected")
    self.removeDbButton.setIcon(slicer.app.style().standardIcon(qt.QStyle.SP_TrashIcon))
    self.removeDbButton.enabled = False
    listButtonLayout = qt.QVBoxLayout()
    listButtonLayout.addWidget(self.loadDbButton)
    listButtonLayout.addWidget(self.removeDbButton)
    listLayout = qt.QHBoxLayout()
    listLayout.addWidget(self.databaseListWidget)
    listLayout.addLayout(listButtonLayout)
    dbLayout.addRow("Available Databases:", listLayout)

    ingestWidget = ctk.ctkCollapsibleButton()
    ingestWidget.text = "Ingest SSM"
    ingestLayout = qt.QFormLayout(ingestWidget)
    self.layout.addWidget(ingestWidget)
    self.ssmTemplateModelSelector = ctk.ctkPathLineEdit()
    self.ssmTemplateModelSelector.filters = ctk.ctkPathLineEdit.Files
    self.ssmTemplateModelSelector.nameFilters=["*.ply", "*.vtk", "*.vtp", "*.stl"]
    ingestLayout.addRow("Template Model:", self.ssmTemplateModelSelector)
    self.ssmTemplateLandmarksSelector = ctk.ctkPathLineEdit()
    self.ssmTemplateLandmarksSelector.filters = ctk.ctkPathLineEdit.Files
    self.ssmTemplateLandmarksSelector.nameFilters=["*.fcsv", "*.json"]
    ingestLayout.addRow("Dense Correspondences:", self.ssmTemplateLandmarksSelector)
    self.ssmSparseLandmarksSelector = ctk.ctkPathLineEdit()
    self.ssmSparseLandmarksSelector.filters = ctk.ctkPathLineEdit.Files
    self.ssmSparseLandmarksSelector.nameFilters = ["*.fcsv", "*.json"]
    ingestLayout.addRow("Sparse Landmarks:", self.ssmSparseLandmarksSelector)
    self.ssmPopulationDirSelector = ctk.ctkPathLineEdit()
    self.ssmPopulationDirSelector.filters = ctk.ctkPathLineEdit.Dirs
    ingestLayout.addRow("Population Correspondences Folder:", self.ssmPopulationDirSelector)
    self.ssmNameEditor = qt.QLineEdit()
    ingestLayout.addRow("New Database Name:", self.ssmNameEditor)
    self.ingestButton = qt.QPushButton("Ingest into Database")
    self.ingestButton.enabled = False
    ingestLayout.addRow(self.ingestButton)


    # --- VISUALIZATION UI ---
    explorerWidget = ctk.ctkCollapsibleButton()
    explorerWidget.text = "SSM Visualization"
    explorerLayout = qt.QFormLayout(explorerWidget)
    self.layout.addWidget(explorerWidget)

    self.explorerModelSelector = slicer.qMRMLNodeComboBox()
    self.explorerModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.explorerModelSelector.selectNodeUponCreation = True
    self.explorerModelSelector.addEnabled = False
    self.explorerModelSelector.removeEnabled = False
    self.explorerModelSelector.noneEnabled = True
    self.explorerModelSelector.setMRMLScene(slicer.mrmlScene)
    self.explorerModelSelector.setToolTip("Select the surface model to deform.")
    explorerLayout.addRow("Model to Deform:", self.explorerModelSelector)

    self.explorerLandmarksSelector = slicer.qMRMLNodeComboBox()
    self.explorerLandmarksSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.explorerLandmarksSelector.selectNodeUponCreation = True
    self.explorerLandmarksSelector.addEnabled = False
    self.explorerLandmarksSelector.removeEnabled = False
    self.explorerLandmarksSelector.noneEnabled = True
    self.explorerLandmarksSelector.setMRMLScene(slicer.mrmlScene)
    self.explorerLandmarksSelector.setToolTip("Select the correspondences that control the deformation.")
    explorerLayout.addRow("Dense Correspondences:", self.explorerLandmarksSelector)

    self.explorerTableSelector = slicer.qMRMLNodeComboBox()
    self.explorerTableSelector.nodeTypes = ["vtkMRMLTableNode"]
    self.explorerTableSelector.addAttribute("vtkMRMLTableNode", "ssm_eigenvalues")
    self.explorerTableSelector.selectNodeUponCreation = True
    self.explorerTableSelector.addEnabled = False
    self.explorerTableSelector.removeEnabled = False
    self.explorerTableSelector.noneEnabled = True
    self.explorerTableSelector.setMRMLScene(slicer.mrmlScene)
    self.explorerTableSelector.setToolTip("Select the SSM data table containing model data.")
    explorerLayout.addRow("SSM Data Table:", self.explorerTableSelector)

    self.slidersWidget = qt.QWidget()
    self.slidersLayout = qt.QFormLayout(self.slidersWidget)
    explorerLayout.addRow(self.slidersWidget)

    self.resetButton = qt.QPushButton("Reset Deformations")
    self.resetButton.toolTip = "Reset all sliders to zero to show the mean shape."
    self.resetButton.enabled = False
    explorerLayout.addRow(self.resetButton)

    self.progressBar = qt.QProgressBar()
    self.progressBar.setRange(0, 100)
    self.progressBar.setValue(0)
    self.progressBar.setFormat("%v% - Ready")
    self.layout.addWidget(self.progressBar)

    # -- Connections --
    self.databasePathSelector.currentPathChanged.connect(self.onDatabasePathChanged)
    self.databaseListWidget.currentItemChanged.connect(self.onDatabaseSelectionChanged)
    self.loadDbButton.connect('clicked(bool)', self.onLoadButton)
    self.removeDbButton.connect('clicked(bool)', self.onRemoveButton)
    self.ssmTemplateModelSelector.connect('validInputChanged(bool)', self.onSelect)
    self.ssmTemplateLandmarksSelector.connect('validInputChanged(bool)', self.onSelect)
    self.ssmSparseLandmarksSelector.connect('validInputChanged(bool)', self.onSelect)
    self.ssmPopulationDirSelector.connect('validInputChanged(bool)', self.onSelect)
    self.ingestButton.connect('clicked(bool)', self.onIngestButton)
    
    self.explorerModelSelector.currentNodeChanged.connect(self.onExplorerNodeSelected)
    self.explorerLandmarksSelector.currentNodeChanged.connect(self.onExplorerNodeSelected)
    self.explorerTableSelector.currentNodeChanged.connect(self.onExplorerNodeSelected)
    self.resetButton.connect('clicked(bool)', self.onResetSliders)

    self.initializeDatabasePath()
    self.layout.addStretch(1)

  def cleanup(self):
    """Called when the user switches to another module."""
    if self.pcSelectorSpinBox:
        self.pcSelectorSpinBox.valueChanged.disconnect(self.onDeformationSliderChanged)
    if self.pcSlider:
        self.pcSlider.valueChanged.disconnect(self.onDeformationSliderChanged)
        
    super().cleanup()

  def onExplorerNodeSelected(self):
    """Called when user selects a different node in the visualization section."""
    self.setupExplorerSliders()
  
  def _clearLayout(self, layout):
    """Removes all widgets from a layout."""
    if not layout: return
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()

  def setupExplorerSliders(self):
    """Creates a UI with a spinbox to select a PC and one slider to control it."""
    self._clearLayout(self.slidersLayout)
    self.ssm_data = {}
    self.pcSelectorSpinBox = None
    self.pcSlider = None
    self.resetButton.enabled = False
    
    modelNode = self.explorerModelSelector.currentNode()
    landmarkNode = self.explorerLandmarksSelector.currentNode()
    tableNode = self.explorerTableSelector.currentNode()

    if not all([modelNode, landmarkNode, tableNode]):
        return

    eigenvalues_str = tableNode.GetAttribute("ssm_eigenvalues")
    if not eigenvalues_str:
        return

    num_points_in_landmarks = landmarkNode.GetNumberOfControlPoints()
    num_rows_in_table = tableNode.GetTable().GetNumberOfRows()
    if num_rows_in_table != (num_points_in_landmarks * 3):
        slicer.util.errorDisplay("Mismatch between number of landmarks and data in SSM table.")
        return
    
    need=int(tableNode.GetAttribute("ssm_npoints") or 0)
    if need and need!=num_points_in_landmarks:
        slicer.util.errorDisplay("SSM table point count does not match landmark count.")
        return

    self.ssm_data['template_vertices'] = slicer.util.arrayFromModelPoints(modelNode)
    self.ssm_data['original_points_RAS'] = slicer.util.arrayFromMarkupsControlPoints(landmarkNode)
    
    mean_flat = slicer.util.arrayFromTableColumn(tableNode, "mean_shape")
    mean_2d = mean_flat.reshape(num_points_in_landmarks, 3)
    self.ssm_data['mean_RAS'] = mean_2d 

    self.ssm_data['modes_RAS'] = []
    num_modes_from_table = tableNode.GetTable().GetNumberOfColumns() - 1
    if num_modes_from_table <= 0:
        slicer.util.infoDisplay("The loaded SSM has no modes of variation.")
        return
    
    for i in range(num_modes_from_table):
        mode_flat = slicer.util.arrayFromTableColumn(tableNode, f"mode_{i}")
        mode_2d = mode_flat.reshape(num_points_in_landmarks, 3)
        self.ssm_data['modes_RAS'].append(mode_2d) 
    
    self.ssm_data['modes_RAS'] = np.stack(self.ssm_data['modes_RAS'], axis=-1)


    # --- pre-computation using K-Nearest Neighbors ---
    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
    slicer.util.showStatusMessage("Building K-NN interpolation map...")

    # The number of nearest landmarks to influence each mesh vertex.
    k_neighbors = 5

    # 1. Build a k-d tree from the mean landmark positions for fast lookups.
    landmark_tree = cKDTree(self.ssm_data['mean_RAS'])

    # 2. For each mesh vertex, find its k-nearest landmarks.
    try:
        distances, indices = landmark_tree.query(self.ssm_data['template_vertices'], k=k_neighbors, workers=-1)  # SciPy>=1.6
    except TypeError:
        distances, indices = landmark_tree.query(self.ssm_data['template_vertices'], k=k_neighbors)

    # 3. Calculate interpolation weights based on inverse distance.
    # Add a small epsilon to avoid division by zero if a vertex is on a landmark.
    weights = 1.0 / (distances + 1e-9)
    # Normalize weights so they sum to 1 for each vertex.
    weights /= np.sum(weights, axis=1, keepdims=True)

    # 4. Store the pre-computed map for real-time use.
    self.ssm_data['interpolation_indices'] = indices
    self.ssm_data['interpolation_weights'] = weights

    slicer.app.restoreOverrideCursor()

    self.ssm_data['eigenvalues'] = np.array(json.loads(eigenvalues_str))
    self.ssm_data['std_devs'] = np.sqrt(self.ssm_data['eigenvalues'])

    self.pcSelectorSpinBox = qt.QSpinBox()
    self.pcSelectorSpinBox.setRange(0, num_modes_from_table - 1)
    self.pcSelectorSpinBox.setToolTip("Select which principal component to visualize.")
    self.slidersLayout.addRow("Principal Component:", self.pcSelectorSpinBox)

    self.pcSlider = ctk.ctkSliderWidget()
    self.pcSlider.minimum = -3
    self.pcSlider.maximum = 3
    self.pcSlider.value = 0
    self.pcSlider.singleStep = 0.1
    self.pcSlider.setToolTip("Deform the shape along the selected PC.")
    self.slidersLayout.addRow("Deformation:", self.pcSlider)
    
    self.pcSelectorSpinBox.valueChanged.connect(self.onDeformationSliderChanged)
    self.pcSlider.valueChanged.connect(self.onDeformationSliderChanged)
    
    self.resetButton.enabled = True
    self.onDeformationSliderChanged()

  def onResetSliders(self):
    """Resets the deformation slider and un-applies the transform."""
    modelNode = self.explorerModelSelector.currentNode()
    if modelNode:
        modelNode.SetAndObserveTransformNodeID(None)
    if modelNode and 'template_vertices' in self.ssm_data:
      points = modelNode.GetPolyData().GetPoints()
      original_points_vtk_array = numpy_support.numpy_to_vtk(self.ssm_data['template_vertices'], deep = 1)
      points.SetData(original_points_vtk_array)
      points.Modified()
        
    if self.pcSlider:
        b = qt.QSignalBlocker(self.pcSlider)
        self.pcSlider.value = 0
        del b  # unblocks
    
    if self.explorerLandmarksSelector.currentNode() and 'original_points_RAS' in self.ssm_data:
        slicer.util.updateMarkupsControlPointsFromArray(
            self.explorerLandmarksSelector.currentNode(), 
            self.ssm_data['original_points_RAS']
        )

  def onDeformationSliderChanged(self):
      modelNode = self.explorerModelSelector.currentNode()
      landmarkNode = self.explorerLandmarksSelector.currentNode()

      if not all([modelNode, landmarkNode, 'interpolation_indices' in self.ssm_data]):
          return

      pc_index = self.pcSelectorSpinBox.value
      slider_value = self.pcSlider.value
      weight = slider_value * self.ssm_data['std_devs'][pc_index]

      # --- Landmark Update ---
      selected_mode_for_landmarks = self.ssm_data['modes_RAS'][:, :, pc_index]
      current_shape_RAS = self.ssm_data['mean_RAS'] + selected_mode_for_landmarks * weight
      slicer.util.updateMarkupsControlPointsFromArray(landmarkNode, current_shape_RAS)

      # --- Mesh Deformation ---
      selected_mode_for_mesh = self.ssm_data['modes_RAS'][:, :, pc_index]
      landmark_displacements = selected_mode_for_mesh * weight

      indices = self.ssm_data['interpolation_indices']
      weights = self.ssm_data['interpolation_weights']

      neighbor_displacements = landmark_displacements[indices]
      vertex_displacements = np.sum(neighbor_displacements * weights[..., np.newaxis], axis=1)

      new_vertices = self.ssm_data['template_vertices'] + vertex_displacements
      
      # 1. Get the vtkPoints object from the model's polydata
      points = modelNode.GetPolyData().GetPoints()

      # 2. Convert the numpy array of new vertex positions to a vtkDataArray
      updated_points_vtk_array = numpy_support.numpy_to_vtk(new_vertices, deep=1)

      # 3. Set the vtkDataArray as the new data for the vtkPoints object
      points.SetData(updated_points_vtk_array)
      
      # 4. Notify VTK that the points have been modified to trigger a re-render
      points.Modified()
      
  def initializeDatabasePath(self):
    settings = qt.QSettings()
    defaultPath = os.path.join(Path.home(), "Documents", "AtlasDatabase")
    dbPath = settings.value('DATABASE/databasePath', defaultPath)
    self.databasePathSelector.setCurrentPath(dbPath)

  def onDatabasePathChanged(self, path):
    if os.path.isdir(path):
        settings = qt.QSettings()
        settings.setValue('DATABASE/databasePath', path)
        self.refreshDatabaseList()

  def refreshDatabaseList(self):
    self.databaseListWidget.clear()
    dbPath = self.databasePathSelector.currentPath
    if not os.path.isdir(dbPath):
        try: os.makedirs(dbPath, exist_ok=True)
        except: pass
        return
    for itemName in sorted(os.listdir(dbPath)):
        if os.path.isdir(os.path.join(dbPath, itemName)):
            self.databaseListWidget.addItem(qt.QListWidgetItem(itemName))

  def onDatabaseSelectionChanged(self):
    isItemSelected = self.databaseListWidget.currentItem() is not None
    self.removeDbButton.enabled = isItemSelected
    self.loadDbButton.enabled = isItemSelected

  def onLoadButton(self):
    currentItem = self.databaseListWidget.currentItem()
    if not currentItem: return
    dbPath = os.path.join(self.databasePathSelector.currentPath, currentItem.text())
    success, message = self.logic.loadSSMDatabase(dbPath)
    if success:
      slicer.util.showStatusMessage(f"✅ {message}", 3000)
      logging.info(message)
    else:
      slicer.util.errorDisplay(f"Error loading database: {message}")

  def onRemoveButton(self):
    currentItem = self.databaseListWidget.currentItem()
    if not currentItem: return
    dbName = currentItem.text()
    if slicer.util.confirmOkCancelDisplay(f"Permanently delete the '{dbName}' database? This cannot be undone."):
      dbPath = os.path.join(self.databasePathSelector.currentPath, dbName)
      success, message = self.logic.removeSSMDatabase(dbPath)
      if success:
        slicer.util.showStatusMessage(f"✅ {message}", 3000)
        self.refreshDatabaseList()
      else:
        slicer.util.errorDisplay(message)

  def onSelect(self):
    self.ingestButton.enabled = bool(
      self.ssmTemplateModelSelector.currentPath and
      self.ssmTemplateLandmarksSelector.currentPath and
      self.ssmSparseLandmarksSelector.currentPath and
      self.ssmPopulationDirSelector.currentPath
    )

  def onIngestButton(self):
    populationDir = self.ssmPopulationDirSelector.currentPath
    if not os.path.isdir(populationDir):
        slicer.util.errorDisplay(f"The specified population folder does not exist:\n\n{populationDir}")
        return

    dbName = self.ssmNameEditor.text or Path(self.ssmTemplateModelSelector.currentPath).stem
    dbRootPath = self.databasePathSelector.currentPath
    db_path = os.path.join(dbRootPath, dbName)

    if os.path.exists(db_path):
        if not slicer.util.confirmOkCancelDisplay(f"Database '{dbName}' already exists. Overwrite?"):
            return

    self.ingestButton.enabled = False
    self.loadDbButton.enabled = False
    self.removeDbButton.enabled = False
    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)

    def progress_callback(value, text):
        self.progressBar.setValue(value)
        self.progressBar.setFormat(f"{text} ({value}%)")
        slicer.app.processEvents() 

    success, message = self.logic.ingestSSMDatabase(
        dbPath=db_path,
        templateModelPath=self.ssmTemplateModelSelector.currentPath,
        templateLandmarksPath=self.ssmTemplateLandmarksSelector.currentPath,
        sparseLandmarksPath=self.ssmSparseLandmarksSelector.currentPath,
        populationDir=populationDir,
        progress_callback=progress_callback
    )

    self.progressBar.setValue(100)
    self.progressBar.setFormat("Complete")
    slicer.app.restoreOverrideCursor()
    self.ingestButton.enabled = True
    self.onDatabaseSelectionChanged()

    if success:
        slicer.util.showStatusMessage(f"✅ Success: {message}", 3000)
        self.refreshDatabaseList()
    else:
        slicer.util.errorDisplay(f"Failed to ingest SSM database:\n\n{message}")
        self.progressBar.setFormat("Failed")


# DATABASETest
#

class DATABASETest(ScriptedLoadableModuleTest):
  def setUp(self): slicer.mrmlScene.Clear(0)
  def runTest(self): self.setUp()