import os, json, shutil, logging
import numpy as np
from pathlib import Path
from vtk.util import numpy_support

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

import slicer, qt, ctk, vtk
from slicer.ScriptedLoadableModule import *

#
# DATABASE
#

class DATABASE(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DATABASE"
    self.parent.categories = ["ATLAS"]
    self.parent.dependencies = []
    self.parent.contributors = ["Arthur Porto"]
    self.parent.helpText = """
    Build a Statistical Shape Model (SSM) and visualize modes of variation on a template.
    Defaults to Local RBF (Gaussian KNN). Optionally use Biharmonic (experimental).
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
      n_shapes,n_pts,dim=shapes.shape
      X=shapes.reshape(n_shapes,-1).astype(np.float64,copy=False)
      mu=X.mean(0); Xc=X-mu
      n,D=Xc.shape
      if n<2 or not np.any(Xc): 
          mean=mu.reshape(n_pts,dim).astype(np.float32,copy=False)
          modes=np.zeros((n_pts,dim,0),np.float32)
          eig=np.zeros((0,),np.float64)
          return mean,modes,eig,0
      if D<=n*2:
          U,s,_=np.linalg.svd(Xc,full_matrices=False)
          eigvals=(s**2)/max(n-1,1)
      else:
          C=(Xc@Xc.T)/max(n-1,1)
          evals,U=np.linalg.eigh(C)
          order=np.argsort(evals)[::-1]
          eigvals=evals[order]
          U=U[:,order]
          s=np.sqrt(np.maximum(eigvals,0.0)*max(n-1,1))
      if eigvals.sum()<=0:
          mean=mu.reshape(n_pts,dim).astype(np.float32,copy=False)
          modes=np.zeros((n_pts,dim,0),np.float32)
          eig=np.zeros((0,),np.float64)
          return mean,modes,eig,0
      var_ratio=eigvals/eigvals.sum()
      if max_modes is not None: num_modes=min(max_modes,len(eigvals))
      elif variance_threshold is None or variance_threshold>=1.0: num_modes=len(eigvals)
      else: num_modes=int(np.searchsorted(np.cumsum(var_ratio),variance_threshold))+1
      scales=np.maximum(s[:num_modes],np.finfo(float).eps)
      modes_flat=(Xc.T@U[:,:num_modes])/scales
      modes=modes_flat.reshape(n_pts,dim,num_modes).astype(np.float32,copy=False)
      mean_shape=mu.reshape(n_pts,dim).astype(np.float32,copy=False)
      return mean_shape,modes,eigvals[:num_modes].astype(np.float64,copy=False),num_modes


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
          if 'CoordinateSystem' in line: cs=self._norm_cs(line.split('=')[-1])
          continue
        parts=line.strip().split(',')
        if len(parts)>=4:
          try: pts.append((float(parts[1]),float(parts[2]),float(parts[3])))
          except Exception as e: logging.warning(f"Could not parse point in file {path}: {e}")
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
      if progress_callback: progress_callback(p, msg)
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
    self.ssm_data = {}             # persistent caches for visualization
    self.pcSliders = []            # list[ctk.ctkSliderWidget]
    self.methodCombo = None        # TPS / Biharmonic
    self.numPCSpin = None          # how many PC sliders to show

  # ---------------- UI ----------------

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # --- DB ---
    dbWidget = ctk.ctkCollapsibleButton(); dbWidget.text = "Shape Modeling Database"
    dbLayout = qt.QFormLayout(dbWidget); self.layout.addWidget(dbWidget)
    self.databasePathSelector = ctk.ctkPathLineEdit(); self.databasePathSelector.filters = ctk.ctkPathLineEdit.Dirs
    dbLayout.addRow("Database Location:", self.databasePathSelector)
    self.databaseListWidget = qt.QListWidget()
    self.loadDbButton = qt.QPushButton("Load Selected"); self.loadDbButton.setIcon(slicer.app.style().standardIcon(qt.QStyle.SP_ArrowUp)); self.loadDbButton.enabled = False
    self.removeDbButton = qt.QPushButton("Remove Selected"); self.removeDbButton.setIcon(slicer.app.style().standardIcon(qt.QStyle.SP_TrashIcon)); self.removeDbButton.enabled = False
    listButtonLayout = qt.QVBoxLayout(); listButtonLayout.addWidget(self.loadDbButton); listButtonLayout.addWidget(self.removeDbButton)
    listLayout = qt.QHBoxLayout(); listLayout.addWidget(self.databaseListWidget); listLayout.addLayout(listButtonLayout)
    dbLayout.addRow("Available Databases:", listLayout)

    ingestWidget = ctk.ctkCollapsibleButton(); ingestWidget.text = "Ingest SSM"
    ingestLayout = qt.QFormLayout(ingestWidget); self.layout.addWidget(ingestWidget)
    self.ssmTemplateModelSelector = ctk.ctkPathLineEdit(); self.ssmTemplateModelSelector.filters = ctk.ctkPathLineEdit.Files; self.ssmTemplateModelSelector.nameFilters=["*.ply","*.vtk","*.vtp","*.stl"]
    self.ssmTemplateLandmarksSelector = ctk.ctkPathLineEdit(); self.ssmTemplateLandmarksSelector.filters = ctk.ctkPathLineEdit.Files; self.ssmTemplateLandmarksSelector.nameFilters=["*.json","*.fcsv"]
    self.ssmSparseLandmarksSelector = ctk.ctkPathLineEdit(); self.ssmSparseLandmarksSelector.filters = ctk.ctkPathLineEdit.Files; self.ssmSparseLandmarksSelector.nameFilters=["*.json","*.fcsv"]
    self.ssmPopulationDirSelector = ctk.ctkPathLineEdit(); self.ssmPopulationDirSelector.filters = ctk.ctkPathLineEdit.Dirs
    self.ssmNameEditor = qt.QLineEdit()
    ingestLayout.addRow("Template Model:", self.ssmTemplateModelSelector)
    ingestLayout.addRow("Dense Correspondences:", self.ssmTemplateLandmarksSelector)
    ingestLayout.addRow("Sparse Landmarks:", self.ssmSparseLandmarksSelector)
    ingestLayout.addRow("Population Correspondences Folder:", self.ssmPopulationDirSelector)
    ingestLayout.addRow("New Database Name:", self.ssmNameEditor)
    self.ingestButton = qt.QPushButton("Ingest into Database"); self.ingestButton.enabled = False
    ingestLayout.addRow(self.ingestButton)

    # --- Visualization ---
    explorerWidget = ctk.ctkCollapsibleButton(); explorerWidget.text = "SSM Visualization"
    explorerLayout = qt.QFormLayout(explorerWidget); self.layout.addWidget(explorerWidget)
    explorerLayout.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)

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

    self.methodCombo = qt.QComboBox()
    self.methodCombo.addItems(["Local RBF (Gaussian, fast)", "Biharmonic (experimental)"])
    self.methodCombo.setCurrentIndex(0)
    explorerLayout.addRow("Deformation Method:", self.methodCombo)

    self.numPCSpin = qt.QSpinBox(); self.numPCSpin.setMinimum(1); self.numPCSpin.setMaximum(1); self.numPCSpin.setValue(2)
    explorerLayout.addRow("Number of PC sliders:", self.numPCSpin)

    # Sliders area — larger and expanding
    self.slidersArea = qt.QScrollArea()
    self.slidersArea.setWidgetResizable(True)
    self.slidersArea.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
    self.slidersArea.setMinimumHeight(420)

    self.slidersWidget = qt.QWidget()
    self.slidersLayout = qt.QFormLayout(self.slidersWidget)
    self.slidersLayout.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    self.slidersArea.setWidget(self.slidersWidget)
    explorerLayout.addRow(self.slidersArea)

    self.resetButton = qt.QPushButton("Reset All PCs")
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

    self.methodCombo.currentIndexChanged.connect(self.onMethodChanged)
    self.numPCSpin.valueChanged.connect(self.rebuildPCSliders)
    self.resetButton.connect('clicked(bool)', self.onResetAll)

    self.initializeDatabasePath()
    self.layout.addStretch(1)

  def cleanup(self):
    for s in self.pcSliders:
      try: s.valueChanged.disconnect(self.onPCSlidersChanged)
      except: pass
    super().cleanup()

  # ------------- DB handlers -------------

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
      dbName = currentItem.text()
      dbPath = os.path.join(self.databasePathSelector.currentPath, dbName)

      success, message = self.logic.loadSSMDatabase(dbPath)
      if success:
        slicer.util.showStatusMessage(f"✅ {message}", 3000)
        logging.info(message)

        # Force default to TPS for newly loaded databases
        b = qt.QSignalBlocker(self.methodCombo)
        try:
          self.methodCombo.setCurrentIndex(0)  # 0
        finally:
          del b

        # Auto-select nodes created by load (this will trigger onExplorerNodeSelected)
        try:
          m = slicer.util.getNode(f"{dbName}_template")
          l = slicer.util.getNode(f"{dbName}_template_correspondences")
          t = slicer.util.getNode(f"ssm_data_{dbName}")
          self.explorerModelSelector.setCurrentNode(m)
          self.explorerLandmarksSelector.setCurrentNode(l)
          self.explorerTableSelector.setCurrentNode(t)
        except Exception as e:
          logging.warning(f"Auto-select failed: {e}")

        # Optional: keep this if you want an explicit rebuild after signals fire
        self.setupExplorerSliders()
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

  # ------------- Viz core -------------

  def _clearLayout(self, layout):
    if not layout: return
    while layout.count():
      child = layout.takeAt(0)
      if child.widget():
        child.widget().deleteLater()

  def _recreateSlidersPanel(self):
    """Replace the sliders panel widget completely (avoids duplicate visual copies)."""
    old = self.slidersArea.takeWidget()
    if old is not None:
      old.deleteLater()
    self.slidersWidget = qt.QWidget()
    self.slidersLayout = qt.QFormLayout(self.slidersWidget)
    self.slidersLayout.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    self.slidersLayout.setVerticalSpacing(12)
    self.slidersArea.setWidget(self.slidersWidget)

  def _methodIndex(self):
    # Slicer/Qt may expose currentIndex as a property (int) or as a callable
    try: return int(self.methodCombo.currentIndex)
    except TypeError: return int(self.methodCombo.currentIndex())

  def onExplorerNodeSelected(self):
    self.setupExplorerSliders()

  def onMethodChanged(self, *args):
    if not self.ssm_data: return
    ok = self.precomputeDeformer()
    if not ok:
      b = qt.QSignalBlocker(self.methodCombo); self.methodCombo.setCurrentIndex(0); del b
      slicer.util.errorDisplay("Biharmonic precompute failed; reverted to Local RBF.")
      self.precomputeDeformer()
    self.applyCurrentPCs()

  def setupExplorerSliders(self):
    # Disconnect any previous sliders and rebuild the panel fresh
    for s in self.pcSliders:
      try: s.valueChanged.disconnect(self.onPCSlidersChanged)
      except: pass
    self.pcSliders = []
    self.ssm_data = {}
    self.resetButton.enabled = False
    self._recreateSlidersPanel()
    
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

    # cache geometry/SSM
    self.ssm_data['template_vertices'] = slicer.util.arrayFromModelPoints(modelNode).astype(np.float64, copy=False)
    self.ssm_data['original_points_RAS'] = slicer.util.arrayFromMarkupsControlPoints(landmarkNode).astype(np.float64, copy=False)
    
    mean_flat = slicer.util.arrayFromTableColumn(tableNode, "mean_shape")
    self.ssm_data['mean_RAS'] = mean_flat.reshape(num_points_in_landmarks, 3).astype(np.float64, copy=False)

    modes = []
    num_modes_from_table = tableNode.GetTable().GetNumberOfColumns() - 1
    if num_modes_from_table <= 0:
      slicer.util.infoDisplay("The loaded SSM has no modes of variation.")
      return
    
    for i in range(num_modes_from_table):
      mode_flat = slicer.util.arrayFromTableColumn(tableNode, f"mode_{i}")
      modes.append(mode_flat.reshape(num_points_in_landmarks, 3))
    self.ssm_data['modes_RAS'] = np.stack(modes, axis=-1).astype(np.float64, copy=False)

    self.ssm_data['eigenvalues'] = np.array(json.loads(eigenvalues_str))
    self.ssm_data['std_devs'] = np.sqrt(self.ssm_data['eigenvalues'])

    # precompute for current method
    ok = self.precomputeDeformer()
    if (not ok) and (self._methodIndex()==1):
      b = qt.QSignalBlocker(self.methodCombo); self.methodCombo.setCurrentIndex(0); del b
      slicer.util.errorDisplay("Biharmonic precompute failed; reverted to TPS.")
      self.precomputeDeformer()

    # build multi-PC sliders (1-indexed)
    self.numPCSpin.setMaximum(num_modes_from_table)
    self.numPCSpin.setValue(min(3, num_modes_from_table))
    self.rebuildPCSliders()

    self.resetButton.enabled = True
    self.applyCurrentPCs()

  def rebuildPCSliders(self):
    # Recreate the panel to guarantee a single fresh set of sliders
    self._recreateSlidersPanel()
    for s in self.pcSliders:
      try: s.valueChanged.disconnect(self.onPCSlidersChanged)
      except: pass
    self.pcSliders = []
    k = self.numPCSpin.value
    for i in range(k):
      sw = ctk.ctkSliderWidget()
      sw.minimum = -3; sw.maximum = 3; sw.value = 0; sw.singleStep = 0.1
      sw.setToolTip(f"PC{i+1} (±σ multiples)")
      self.slidersLayout.addRow(f"PC{i+1}:", sw)   # 1-indexed label
      sw.valueChanged.connect(self.onPCSlidersChanged)
      self.pcSliders.append(sw)

  def onResetAll(self):
    # Reset sliders and geometry; avoid context-manager (not supported)
    for s in self.pcSliders:
      blocker = qt.QSignalBlocker(s); s.value = 0; del blocker
    self.resetGeometry()

  def resetGeometry(self):
    modelNode = self.explorerModelSelector.currentNode()
    if modelNode and 'template_vertices' in self.ssm_data:
      pd = modelNode.GetPolyData()
      pts = pd.GetPoints()
      original_points_vtk_array = numpy_support.numpy_to_vtk(self.ssm_data['template_vertices'].astype(np.float32, copy=False), deep=1)
      pts.SetData(original_points_vtk_array)
      pts.Modified(); pd.Modified(); modelNode.Modified()
    if self.explorerLandmarksSelector.currentNode() and 'original_points_RAS' in self.ssm_data:
      slicer.util.updateMarkupsControlPointsFromArray(self.explorerLandmarksSelector.currentNode(), self.ssm_data['original_points_RAS'])

  def onPCSlidersChanged(self, *args):
    self.applyCurrentPCs()

  def _currentPCWeights(self):
    if not self.ssm_data or not self.pcSliders: return None
    m = self.ssm_data['modes_RAS'].shape[-1]
    k = min(len(self.pcSliders), m, len(self.ssm_data['std_devs']))
    w = np.zeros(m, dtype=np.float64)
    # each slider is in units of σ for PC i (1-indexed)
    for i in range(k):
      w[i] = float(self.pcSliders[i].value) * float(self.ssm_data['std_devs'][i])
    return w

  def applyCurrentPCs(self):
    modelNode = self.explorerModelSelector.currentNode()
    landmarkNode = self.explorerLandmarksSelector.currentNode()
    if not (modelNode and landmarkNode and self.ssm_data):
      return
    w = self._currentPCWeights()
    if w is None:
      return
    modes = self.ssm_data['modes_RAS']
    mean = self.ssm_data['mean_RAS']
    # aggregate deformation across first k PCs shown
    k = min(len(self.pcSliders), modes.shape[-1])
    agg = np.tensordot(modes[...,:k], w[:k], axes=(2,0))   # (n_pts,3)
    target = mean + agg

    # Update landmarks
    slicer.util.updateMarkupsControlPointsFromArray(landmarkNode, target.astype(np.float64, copy=False))

    # Update mesh by selected method
    if self._methodIndex()==0:
      self._applyLocalRBF(target)
    else:
      ok = self._applyBiharmonic(target)
      if not ok:
        blocker = qt.QSignalBlocker(self.methodCombo); self.methodCombo.setCurrentIndex(0); del blocker
        self.precomputeDeformer()
        self._applyLocalRBF(target)

  # -------- Precomputes & solvers --------

  def precomputeDeformer(self):
      try:
          modelNode = self.explorerModelSelector.currentNode()
          if modelNode is None: return False

          V = slicer.util.arrayFromModelPoints(modelNode).astype(np.float64, copy=False)
          self.ssm_data['template_vertices'] = V

          if self._methodIndex()==0:
              lm = self.ssm_data['mean_RAS'].astype(np.float64, copy=False)
              k = min(32, lm.shape[0])
              tree = cKDTree(lm)
              try: d, idx = tree.query(V, k=k, workers=-1)
              except TypeError: d, idx = tree.query(V, k=k)
              h = np.percentile(d[:, -1], 75) + 1e-9
              w = np.exp(-(d*d)/(h*h))
              w /= w.sum(1, keepdims=True)
              self.ssm_data['interpolation_indices'] = idx.astype(np.int64, copy=False)
              self.ssm_data['interpolation_weights']  = w.astype(np.float64, copy=False)
              return True

          pd = modelNode.GetPolyData()

          def triangleize_with_ids(pd_in):
              tf = vtk.vtkTriangleFilter(); tf.SetInputData(pd_in); tf.PassLinesOff(); tf.PassVertsOff(); tf.Update()
              tri = tf.GetOutput()
              if tri.GetNumberOfPoints() == pd_in.GetNumberOfPoints():
                  return tri, None
              idf = vtk.vtkIdFilter(); idf.SetInputData(pd_in); idf.PointIdsOn()
              if hasattr(idf,'SetPointIdsArrayName'): idf.SetPointIdsArrayName('orig_pid')
              elif hasattr(idf,'SetIdsArrayName'): idf.SetIdsArrayName('orig_pid')
              idf.CellIdsOff(); idf.Update()
              tf2 = vtk.vtkTriangleFilter(); tf2.SetInputConnection(idf.GetOutputPort()); tf2.PassLinesOff(); tf2.PassVertsOff(); tf2.Update()
              tri2 = tf2.GetOutput()
              arr = tri2.GetPointData().GetArray('orig_pid')
              orig_pid = numpy_support.vtk_to_numpy(arr).astype(np.int64, copy=False) if arr else None
              return tri2, orig_pid

          tri, orig_pid = triangleize_with_ids(pd)
          Vtri = numpy_support.vtk_to_numpy(tri.GetPoints().GetData()).astype(np.float64, copy=False)
          ca = numpy_support.vtk_to_numpy(tri.GetPolys().GetData())
          F = ca.reshape(-1,4)[:,1:4].astype(np.int64, copy=False)

          n = Vtri.shape[0]
          I=[]; J=[]; W=[]; A=np.zeros(n, dtype=np.float64)
          for t in F:
              i,j,k = int(t[0]),int(t[1]),int(t[2])
              vi,vj,vk = Vtri[i],Vtri[j],Vtri[k]
              e0 = vj-vk; e1 = vk-vi; e2 = vi-vj
              nrm = np.linalg.norm(np.cross(e0,-e2))
              if nrm<=0: continue
              c0 = -np.dot(e0,e2)/nrm
              c1 = -np.dot(e1,e0)/nrm
              c2 = -np.dot(e2,e1)/nrm
              Atri = 0.5*nrm
              A[i]+=Atri/3; A[j]+=Atri/3; A[k]+=Atri/3
              for (a,b,cot) in [(j,k,c0),(k,i,c1),(i,j,c2)]:
                  I.append(a); J.append(b); W.append(cot/2.0)
                  I.append(b); J.append(a); W.append(cot/2.0)
          Lij = coo_matrix((W,(I,J)),shape=(n,n)).tocsr()
          d = -np.array(Lij.sum(axis=1)).ravel()
          L = Lij.copy(); L.setdiag(d)
          M = diags(np.maximum(A,1e-12))
          Minv = diags(1.0/np.maximum(M.diagonal(),1e-12))
          A0 = (L.T @ Minv @ L).tocsr()

          lm_mean = self.ssm_data['mean_RAS'].astype(np.float64, copy=False)
          kdt = cKDTree(Vtri); _, idx = kdt.query(lm_mean, k=1)

          Wd = np.zeros(n, dtype=np.float64); Wd[idx] = 1.0
          lam = 1e4
          diag_max = float(A0.diagonal().max() or 1.0)
          if lam < 1e-6*diag_max or lam > 1e6*diag_max: lam = 1e-2*diag_max

          A = (A0 + lam*diags(Wd)).tocsc()
          lu = splu(A)

          self.ssm_data['bih_V'] = Vtri
          self.ssm_data['constrained_idx'] = idx.astype(np.int64, copy=False)
          self.ssm_data['bih_A_lu'] = lu
          self.ssm_data['bih_lam'] = lam
          self.ssm_data['bih_Wd'] = Wd
          self.ssm_data['faces'] = F
          self.ssm_data['bih_orig_pid'] = orig_pid
          return True
      except Exception as e:
          logging.error(f"Precompute failed: {e}")
          return False

  def _applyLocalRBF(self, target):
    modelNode = self.explorerModelSelector.currentNode()
    V0 = self.ssm_data['template_vertices']
    indices = self.ssm_data['interpolation_indices']
    weights = self.ssm_data['interpolation_weights']
    # total landmark delta across multiple PCs
    delta_landmarks = (target - self.ssm_data['mean_RAS'])  # (nL,3)
    neighbor_deltas = delta_landmarks[indices]             # (nV,k,3)
    disp = np.sum(neighbor_deltas * weights[..., np.newaxis], axis=1)
    new_vertices = V0 + disp

    pd = modelNode.GetPolyData()
    points = pd.GetPoints()
    points.SetData(numpy_support.numpy_to_vtk(new_vertices.astype(np.float32, copy=False), deep=1))
    points.Modified(); pd.Modified(); modelNode.Modified()

  def _applyBiharmonic(self, target):
      try:
          modelNode = self.explorerModelSelector.currentNode()
          Vn = self.ssm_data['bih_V']; n = Vn.shape[0]
          idx = self.ssm_data['constrained_idx']; lu = self.ssm_data['bih_A_lu']; Wd = self.ssm_data['bih_Wd']; lam = self.ssm_data['bih_lam']
          b = np.zeros((n,3), np.float64); b[idx,:] = target
          B = (lam*Wd)[:,None]*b
          X = np.empty_like(b)
          X[:,0] = lu.solve(B[:,0]); X[:,1] = lu.solve(B[:,1]); X[:,2] = lu.solve(B[:,2])

          pd = modelNode.GetPolyData(); pts = pd.GetPoints()
          orig_pid = self.ssm_data.get('bih_orig_pid', None)
          if orig_pid is not None:
              base = slicer.util.arrayFromModelPoints(modelNode).astype(np.float32, copy=False)
              base[orig_pid] = X.astype(np.float32, copy=False)
              arr = numpy_support.numpy_to_vtk(base, deep=1)
          else:
              arr = numpy_support.numpy_to_vtk(X.astype(np.float32, copy=False), deep=1)
          pts.SetData(arr); pts.Modified(); pd.Modified(); modelNode.Modified()
          return True
      except Exception as e:
          logging.error(f"Biharmonic solve failed: {e}")
          return False



# DATABASETest
#

class DATABASETest(ScriptedLoadableModuleTest):
  def setUp(self): slicer.mrmlScene.Clear(0)
  def runTest(self): self.setUp()
