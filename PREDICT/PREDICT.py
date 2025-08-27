import os, logging, copy, json, threading, importlib, numpy as np
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import vtk.util.numpy_support as vtk_np
from scipy.spatial import cKDTree
from scipy.optimize import minimize, differential_evolution

# ---------- Small utilities----------
COLORS = {"red":(1,0.2,0.2), "green":(0.2,1,0.2), "blue":(0.2,0.2,1)}

def np_to_vtk_mat(M):
    M = np.asarray(M, float); assert M.shape==(4,4)
    vm = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4): vm.SetElement(i,j,float(M[i,j]))
    return vm

def vtk_mat_to_np(vm):
    return np.array([[vm.GetElement(i,j) for j in range(4)] for i in range(4)], float)

def as_np4x4(M):
    if isinstance(M, vtk.vtkMatrix4x4): return vtk_mat_to_np(M)
    A = np.asarray(M, float)
    return A if A.shape==(4,4) else np.eye(4, dtype=float)

def make_transform_node(M, name):
    t = vtk.vtkTransform(); t.SetMatrix(np_to_vtk_mat(as_np4x4(M)))
    tn = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', name)
    tn.SetAndObserveTransformToParent(t); return tn

def transform_polydata(pd, M):
    tf = vtk.vtkTransformPolyDataFilter(); tf.SetTransform(vtk.vtkTransform())
    tf.GetTransform().SetMatrix(np_to_vtk_mat(as_np4x4(M)))
    tf.SetInputData(pd); tf.Update(); out = vtk.vtkPolyData(); out.DeepCopy(tf.GetOutput()); return out

def bounds_diag(node_or_pd):
    pd = node_or_pd if isinstance(node_or_pd, vtk.vtkPolyData) else node_or_pd.GetPolyData()
    b = np.array(pd.GetBounds(), float).reshape(3,2)
    return float(np.linalg.norm(b[:,1]-b[:,0]))

# ---------- Module ----------
class PREDICT(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "PREDICT"
    self.parent.categories = ["ATLAS"]
    self.parent.dependencies = []
    self.parent.contributors = ["Arthur Porto"]
    self.parent.helpText = (
      """
          ATLAS — Automated Template-based Landmark Alignment System — 
          is a two-stage landmark transfer and morphometric alignment tool. It combines global rigid registration 
          with Statistical Shape Model–guided Coherent Point Drift for non-rigid alignment, 
          followed by an optional surface projection refinement to ensure landmarks lie precisely on target meshes. 
          PREDICT supports both single-specimen optimization and batch landmark transfer, allowing consistent application 
          of tuned alignment parameters across large datasets.
          <p>For more information see the 
          <a href=\"https://github.com/SlicerMorph/SlicerMorph/tree/master/Docs/PREDICT\">online documentation</a>.</p>
      """
    )
    self.parent.acknowledgementText = "This module was developed by Arthur Porto"

# ---------- Widget ----------
class PREDICTWidget(ScriptedLoadableModuleWidget):
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.cloneFolderItemID = None
    self._deps_ready = False
    qt.QTimer.singleShot(0, lambda: self._ensure_deps_async())

  # ----- Dependencies installer (async-safe) -----
  def _ensure_deps_async(self, on_ready=None):
    import importlib.util, traceback, sys, subprocess
    required=[("tiny3d","tiny3d"),("biocpd","biocpd")]
    missing=[m for m,_ in required if importlib.util.find_spec(m) is None]
    if missing:
        msg="PREDICT needs: "+", ".join(missing)+".\nInstall now?"
        if not slicer.util.confirmOkCancelDisplay(msg):
            slicer.util.errorDisplay("Dependencies not installed; some actions may fail."); return
    self._deps_ready=False; self._deps_error=None
    def worker():
        try:
            if missing:
                specs=[spec for m,spec in required if m in missing]
                subprocess.check_call([sys.executable,"-m","pip","install",*specs])
            for m in ("tiny3d","biocpd","scipy.spatial","scipy.optimize"):
                importlib.import_module(m)
        except Exception as e:
            self._deps_error=(e, traceback.format_exc())
        else:
            self._deps_ready=True
    threading.Thread(target=worker, daemon=True).start()
    def poll():
        if self._deps_error:
            e,tb=self._deps_error; self._deps_error=None
            slicer.util.errorDisplay(f"PREDICT install failed:\n{e}\n{tb}"); return
        if self._deps_ready:
            slicer.util.showStatusMessage("PREDICT: dependencies ready", 3000)
            if callable(on_ready): on_ready(); return
        qt.QTimer.singleShot(150, poll)
    qt.QTimer.singleShot(0, poll)

  # ----- Parameter schema (declarative Advanced tab) -----
  PARAMS = [
    {"key":"skipScaling","kind":"check","label":"Skip scaling","section":"General Settings","value":False},
    {"key":"skipProjection","kind":"check","label":"Skip projection","section":"General Settings","value":False},
    {"key":"skipOptimization","kind":"check","label":"Skip template optimization (batch)","section":"General Settings","value":False},
    {"key":"pointDensity","kind":"slider","label":"Point Density","section":"Point density and max projection","min":0.1,"max":3.0,"step":0.1,"value":1.0},
    {"key":"projectionFactor","kind":"slider","label":"Max projection factor (%)","section":"Point density and max projection","min":0,"max":10,"step":1.0,"value":1.0},
    {"key":"normalSearchRadius","kind":"slider","label":"Normal search radius","section":"Rigid registration","min":2,"max":12,"step":1,"value":2},
    {"key":"FPFHSearchRadius","kind":"slider","label":"FPFH search radius","section":"Rigid registration","min":3,"max":20,"step":1,"value":5},
    {"key":"distanceThreshold","kind":"slider","label":"RANSAC distance threshold","section":"Rigid registration","min":0.5,"max":4.0,"step":0.25,"value":1.5},
    {"key":"maxRANSAC","kind":"spin","label":"Max RANSAC iterations","section":"Rigid registration","min":1,"max":50_000_000,"step":1,"value":400_000},
    {"key":"confidence","kind":"dspin","label":"RANSAC confidence","section":"Rigid registration","min":0.0,"max":1.0,"step":0.001,"value":0.999},
    {"key":"ICPDistanceThreshold","kind":"slider","label":"ICP distance threshold","section":"Rigid registration","min":0.1,"max":2.0,"step":0.1,"value":0.4},
    {"key":"alpha","kind":"dspin","label":"Rigidity (alpha)","section":"PCA-CPD registration","min":0.1,"max":10.0,"step":0.1,"value":2.0},
    {"key":"beta","kind":"dspin","label":"Motion coherence (beta)","section":"PCA-CPD registration","min":0.1,"max":10.0,"step":0.1,"value":2.0},
    {"key":"w","kind":"dspin","label":"Outlier weight (w)","section":"PCA-CPD registration","min":0.0,"max":0.5,"step":0.01,"value":0.10,"decimals":3},
    {"key":"tolerance","kind":"dspin","label":"Tolerance","section":"PCA-CPD registration","min":1e-8,"max":1e-2,"step":1e-6,"value":1e-6,"decimals":8},
    {"key":"max_iterations","kind":"spin","label":"Max iterations","section":"PCA-CPD registration","min":100,"max":1000,"step":50,"value":250},
  ]

  def _make_selector(self, types, attr_key=None, attr_val=None, tooltip=None, none=False):
    sel = slicer.qMRMLNodeComboBox()
    sel.nodeTypes = types; sel.selectNodeUponCreation=True
    sel.addEnabled=False; sel.removeEnabled=False; sel.noneEnabled=bool(none)
    sel.setMRMLScene(slicer.mrmlScene)
    if attr_key:
      if attr_val is None:
        sel.addAttribute(types[0], attr_key)
      else:
        sel.addAttribute(types[0], attr_key, attr_val)
    if tooltip: sel.setToolTip(tooltip)
    return sel

  def _build_advanced_tab(self, layout):
    sections={}
    def section(name):
      if name not in sections:
        cb=ctk.ctkCollapsibleButton(); cb.text=name; layout.addRow(cb); sections[name]=qt.QFormLayout(cb)
      return sections[name]
    for p in self.PARAMS:
      kind=p["kind"]; lab=p["label"]; key=p["key"]; sec=p["section"]; v=p.get("value")
      if kind=="check":
        w=qt.QCheckBox(); w.checked=bool(v); w.toggled.connect(self._on_param_changed)
      elif kind=="slider":
        w=ctk.ctkSliderWidget(); w.minimum=p["min"]; w.maximum=p["max"]; w.singleStep=p["step"]; w.value=float(v); w.valueChanged.connect(self._on_param_changed)
      elif kind=="spin":
        w=qt.QSpinBox(); w.minimum=int(p["min"]); w.maximum=int(p["max"]); w.singleStep=int(p["step"]); w.value=int(v); w.valueChanged.connect(self._on_param_changed)
      elif kind=="dspin":
        w=ctk.ctkDoubleSpinBox(); w.minimum=float(p["min"]); w.maximum=float(p["max"]); w.singleStep=float(p["step"]); w.value=float(v); w.setDecimals(int(p.get("decimals",3))); w.valueChanged.connect(self._on_param_changed)
      else:
        continue
      setattr(self, f"param_{key}", w)
      section(sec).addRow(lab+": ", w)
    self.parameterDictionary = self._read_params()

  def _read_params(self):
    d={}
    for p in self.PARAMS:
      w=getattr(self, f"param_{p['key']}")
      if p["kind"]=="check": d[p['key']] = bool(w.checked)
      else: d[p['key']] = float(w.value) if hasattr(w,'value') else w.value()
    # integers for specific keys
    d["maxRANSAC"] = int(d["maxRANSAC"]); d["max_iterations"] = int(d["max_iterations"]) 
    d["normalSearchRadius"] = int(d["normalSearchRadius"]); d["FPFHSearchRadius"] = int(d["FPFHSearchRadius"])
    return d

  def _on_param_changed(self, *args):
    self.parameterDictionary = self._read_params()

  # ----- UI setup -----
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    tabsWidget = qt.QTabWidget(); self.layout.addWidget(tabsWidget)
    alignSingleTab=qt.QWidget(); alignMultiTab=qt.QWidget(); advancedSettingsTab=qt.QWidget(); d2Tab=qt.QWidget()
    tabsWidget.addTab(alignSingleTab, "Single Alignment")
    tabsWidget.addTab(alignMultiTab, "Batch processing")
    tabsWidget.addTab(advancedSettingsTab, "Advanced Settings")
    tabsWidget.addTab(d2Tab, "Template Optimization")
    alignSingleTabLayout = qt.QFormLayout(alignSingleTab)
    alignMultiTabLayout  = qt.QFormLayout(alignMultiTab)
    advancedSettingsTabLayout = qt.QFormLayout(advancedSettingsTab)
    d2Layout = qt.QFormLayout(d2Tab)

    # --- Single Alignment ---
    single=ctk.ctkCollapsibleButton(); single.text="Align and subsample a source and target mesh"; singleL=qt.QFormLayout(single)
    alignSingleTabLayout.addRow(single)
    self.sourceModelSelector            = self._make_selector(["vtkMRMLModelNode"]) ; singleL.addRow("Template model:", self.sourceModelSelector)
    self.sourceFiducialSelector         = self._make_selector(["vtkMRMLMarkupsFiducialNode"]) ; singleL.addRow("Template Correspondences:", self.sourceFiducialSelector)
    self.sourceSparseFiducialSelector   = self._make_selector(["vtkMRMLMarkupsFiducialNode"]) ; singleL.addRow("Template Landmarks:", self.sourceSparseFiducialSelector)
    self.targetModelSelector            = self._make_selector(["vtkMRMLModelNode"]) ; singleL.addRow("Target model:", self.targetModelSelector)
    self.ssmTableSelector               = self._make_selector(["vtkMRMLTableNode"], attr_key="ssm_eigenvalues", attr_val=None, tooltip="If you have an SSM table loaded, select it here to use PCA-CPD.") ; singleL.addRow("SSM Data Table:", self.ssmTableSelector)
    self.subsampleButton=qt.QPushButton("Run subsampling"); self.subsampleButton.enabled=False; singleL.addRow(self.subsampleButton)
    self.subsampleInfo=qt.QPlainTextEdit(); self.subsampleInfo.setPlaceholderText("Subsampling information"); self.subsampleInfo.setReadOnly(True); singleL.addRow(self.subsampleInfo)
    self.alignButton=qt.QPushButton("Run rigid step"); self.alignButton.enabled=False; singleL.addRow(self.alignButton)
    self.displayMeshButton=qt.QPushButton("Display alignment"); self.displayMeshButton.enabled=False; singleL.addRow(self.displayMeshButton)
    self.CPDRegistrationButton=qt.QPushButton("Run deformable step"); self.CPDRegistrationButton.enabled=False; singleL.addRow(self.CPDRegistrationButton)
    self.displayWarpedModelButton=qt.QPushButton("Show registration"); self.displayWarpedModelButton.enabled=False; singleL.addRow(self.displayWarpedModelButton)
    self.resetButton=qt.QPushButton("Reset"); singleL.addRow(self.resetButton)

    # --- Batch Processing ---
    multi=ctk.ctkCollapsibleButton(); multi.text="Transfer landmark points from a source mesh to a directory of target meshes"; multiL=qt.QFormLayout(multi)
    alignMultiTabLayout.addRow(multi)
    self.sourceModelMultiSelector          = self._make_selector(["vtkMRMLModelNode"]) ; multiL.addRow("Source mesh:", self.sourceModelMultiSelector)
    self.sourceFiducialMultiSelector       = self._make_selector(["vtkMRMLMarkupsFiducialNode"]) ; multiL.addRow("Source Correspondences:", self.sourceFiducialMultiSelector)
    self.sourceSparseFiducialMultiSelector = self._make_selector(["vtkMRMLMarkupsFiducialNode"]) ; multiL.addRow("Source landmarks:", self.sourceSparseFiducialMultiSelector)
    self.targetModelMultiSelector=ctk.ctkPathLineEdit(); self.targetModelMultiSelector.filters=ctk.ctkPathLineEdit.Dirs; multiL.addRow("Target mesh directory:", self.targetModelMultiSelector)
    self.landmarkOutputSelector=ctk.ctkPathLineEdit(); self.landmarkOutputSelector.filters=ctk.ctkPathLineEdit.Dirs; multiL.addRow("Target output landmark directory:", self.landmarkOutputSelector)
    self.ssmTableMultiSelector = self._make_selector(["vtkMRMLTableNode"], attr_key="ssm_eigenvalues", attr_val=None, tooltip="Select the SSM table.", none=False); multiL.addRow("SSM Data Table:", self.ssmTableMultiSelector)
    self.skipOptBatchChk = qt.QCheckBox("Skip template optimization"); self.skipOptBatchChk.setToolTip("Don’t run SSM grid+RANSAC per target before rigid/CPD.");multiL.addRow(self.skipOptBatchChk)
    self.applyLandmarkMultiButton=qt.QPushButton("Run auto-landmarking"); self.applyLandmarkMultiButton.enabled=False; multiL.addRow(self.applyLandmarkMultiButton)
    self.batchProgress=qt.QProgressBar(); self.batchProgress.minimum=0; self.batchProgress.maximum=100; self.batchProgress.value=0; multiL.addRow("Progress:", self.batchProgress)
    self.batchCancelButton=qt.QPushButton("Cancel batch"); self.batchCancelButton.enabled=False; multiL.addRow(self.batchCancelButton)
    self._cancelBatch=False; self.batchCancelButton.connect('clicked()', self.onCancelBatch)


    # --- Optimization Tab ---
    opt=ctk.ctkCollapsibleButton(); opt.text="Optimize the template"; optL=qt.QFormLayout(opt); d2Layout.addRow(opt)
    self.d2sourceModelSelector          = self._make_selector(["vtkMRMLModelNode"]); optL.addRow("Template model:", self.d2sourceModelSelector)
    self.d2sourceFiducialSelector       = self._make_selector(["vtkMRMLMarkupsFiducialNode"]); optL.addRow("Template Correspondences:", self.d2sourceFiducialSelector)
    self.d2sourceSparseFiducialSelector = self._make_selector(["vtkMRMLMarkupsFiducialNode"]); optL.addRow("Template Landmarks:", self.d2sourceSparseFiducialSelector)
    self.d2targetModelSelector          = self._make_selector(["vtkMRMLModelNode"]); optL.addRow("Target model:", self.d2targetModelSelector)
    self.d2ssmTableSelector             = self._make_selector(["vtkMRMLTableNode"], attr_key="ssm_eigenvalues", attr_val=None); optL.addRow("SSM Data Table:", self.d2ssmTableSelector)
    self.pcGridSteps=qt.QSpinBox(); self.pcGridSteps.minimum=2; self.pcGridSteps.maximum=9; self.pcGridSteps.value=3; self.pcGridSteps.setToolTip("Grid steps per PC"); optL.addRow("Grid steps / PC:", self.pcGridSteps)
    self.ransacItersPerCand=qt.QSpinBox(); self.ransacItersPerCand.minimum=10000; self.ransacItersPerCand.maximum=2000000; self.ransacItersPerCand.singleStep=10000; self.ransacItersPerCand.value=150000; optL.addRow("RANSAC iters (per candidate):", self.ransacItersPerCand)
    self.optimizeButton=qt.QPushButton("Optimize"); optL.addRow(self.optimizeButton)

    # --- Advanced Settings (declarative) ---
    self._build_advanced_tab(advancedSettingsTabLayout)

    
    # Sync Batch checkbox <-> Advanced param widget
    self.skipOptBatchChk.setChecked(bool(self.param_skipOptimization.checked))
    self.skipOptBatchChk.toggled.connect(
        lambda v: (setattr(self.param_skipOptimization, "checked", v), self._on_param_changed())
    )
    self.param_skipOptimization.toggled.connect(
        lambda v: self.skipOptBatchChk.setChecked(v)
    )

    # --- Signals ---
    self.sourceModelSelector.currentNodeChanged.connect(self.onSelect)
    self.sourceFiducialSelector.currentNodeChanged.connect(self.onSelect)
    self.sourceSparseFiducialSelector.currentNodeChanged.connect(self.onSelect)
    self.targetModelSelector.currentNodeChanged.connect(self.onSelect)
    self.ssmTableSelector.currentNodeChanged.connect(self.onSelect)
    self.subsampleButton.clicked.connect(self.onSubsampleButton)
    self.alignButton.clicked.connect(self.onAlignButton)
    self.displayMeshButton.clicked.connect(self.onDisplayMeshButton)
    self.CPDRegistrationButton.clicked.connect(self.onCPDRegistration)
    self.displayWarpedModelButton.clicked.connect(self.onDisplayWarpedModel)
    self.resetButton.clicked.connect(self.onResetButton)

    self.sourceModelMultiSelector.currentNodeChanged.connect(self.onSelectMultiProcess)
    self.sourceFiducialMultiSelector.currentNodeChanged.connect(self.onSelectMultiProcess)
    self.sourceSparseFiducialMultiSelector.currentNodeChanged.connect(self.onSelectMultiProcess)
    self.targetModelMultiSelector.validInputChanged.connect(self.onSelectMultiProcess)
    self.landmarkOutputSelector.validInputChanged.connect(self.onSelectMultiProcess)
    self.applyLandmarkMultiButton.clicked.connect(self.onApplyLandmarkMulti)
    self.ssmTableMultiSelector.currentNodeChanged.connect(self.onSelectMultiProcess)

    self.optimizeButton.clicked.connect(self.onOptimize)

    self.onSelect(); self.parameterDictionary=self._read_params()
    self.layout.addStretch(1)

  # ----- Small helpers -----
  def updateLayout(self):
    lm=slicer.app.layoutManager(); lm.setLayout(9)
    lm.threeDWidget(0).threeDView().resetFocalPoint(); lm.threeDWidget(0).threeDView().resetCamera()

  def _enable_single(self):
    ready = all([self.sourceModelSelector.currentNode(), self.targetModelSelector.currentNode(),
                 self.sourceFiducialSelector.currentNode(), self.sourceSparseFiducialSelector.currentNode()])
    self.subsampleButton.enabled = ready

  def _enable_batch(self):
    ready = all([self.sourceModelMultiSelector.currentNode(), self.sourceFiducialMultiSelector.currentNode(),
                 self.sourceSparseFiducialMultiSelector.currentNode(), self.ssmTableMultiSelector.currentNode(),
                 self.targetModelMultiSelector.currentPath, self.landmarkOutputSelector.currentPath])
    self.applyLandmarkMultiButton.enabled = ready

  def ensureCloneFolder(self, label=None):
    shNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    if not shNode: raise RuntimeError("SubjectHierarchy node not found")
    invalid = shNode.GetInvalidItemID() if hasattr(shNode, "GetInvalidItemID") else -1
    rootName = "PREDICT runs"; rootID = shNode.GetItemByName(rootName)
    if rootID in (None, invalid): rootID = shNode.CreateFolderItem(shNode.GetSceneItemID(), rootName)
    if getattr(self, "cloneFolderItemID", None) not in (None, invalid): return self.cloneFolderItemID
    from datetime import datetime
    runName = f"Run {datetime.now().strftime('%Y%m%d-%H%M%S')}"; 
    if label: runName += f" - {label}"
    self.cloneFolderItemID = shNode.CreateFolderItem(rootID, runName)
    return self.cloneFolderItemID

  def cloneNode(self, originalModelNode, customName=None):
    shNode  = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    shLogic = slicer.modules.subjecthierarchy.logic()
    originalItemID = shNode.GetItemByDataNode(originalModelNode)
    newItemID      = shLogic.CloneSubjectHierarchyItem(shNode, originalItemID)
    cloneNode      = shNode.GetItemDataNode(newItemID)
    cloneNode.SetName(customName or (originalModelNode.GetName()+"_clone"))
    self.ensureCloneFolder(); shNode.SetItemParent(newItemID, self.cloneFolderItemID)
    return cloneNode

  def _parentNode(self, mrmlNode):
    shNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    if not shNode: raise RuntimeError("SubjectHierarchy node not found")
    self.ensureCloneFolder(); invalid = shNode.GetInvalidItemID() if hasattr(shNode, "GetInvalidItemID") else -1
    itemID = shNode.GetItemByDataNode(mrmlNode)
    if itemID == invalid or itemID is None: shNode.CreateItem(self.cloneFolderItemID, mrmlNode)
    else: shNode.SetItemParent(itemID, self.cloneFolderItemID)

  def clearAllRuns(self):
    shNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    if not shNode: return
    itemID = shNode.GetItemByName("PREDICT runs")
    if itemID: shNode.RemoveItem(itemID)
    self.cloneFolderItemID = None

  def clearCloneFolder(self):
    shNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    if not shNode: return
    if getattr(self, "cloneFolderItemID", None): shNode.RemoveItem(self.cloneFolderItemID)
    self.cloneFolderItemID = None

  def _hideInputNodes(self):
    for selector in (self.sourceModelSelector, self.sourceFiducialSelector, self.sourceSparseFiducialSelector, self.targetModelSelector):
      node = selector.currentNode()
      if node and node.GetDisplayNode(): node.GetDisplayNode().SetVisibility(False)

  def _showInputNodes(self, visible=True):
    for selector in (self.sourceModelSelector, self.sourceFiducialSelector, self.sourceSparseFiducialSelector, self.targetModelSelector):
      n = selector.currentNode()
      if n and n.GetDisplayNode(): n.GetDisplayNode().SetVisibility(visible)

  def onSelect(self, node=None):
    src = self.sourceModelSelector.currentNode(); tgt = self.targetModelSelector.currentNode()
    lm  = self.sourceSparseFiducialSelector.currentNode(); slm = self.sourceFiducialSelector.currentNode()
    self._enable_single()
    if src: self.sourceModelMultiSelector.setCurrentNode(src)
    if lm: self.sourceSparseFiducialMultiSelector.setCurrentNode(lm)
    if slm: self.sourceFiducialMultiSelector.setCurrentNode(slm)

  def onCancelBatch(self):
    self._cancelBatch = True; self.batchCancelButton.enabled = False
    slicer.util.showStatusMessage("Cancelling batch…", 2000)

  def onSelectMultiProcess(self): self._enable_batch()

  def _proj_frac(self):
    d=self.parameterDictionary; return 0.0 if d.get("skipProjection", False) else float(d.get("projectionFactor",1.0))/100.0

  # ----- Single alignment flow -----
  def onSubsampleButton(self):
    logic = PREDICTLogic(); self._hideInputNodes()
    srcOrig = self.sourceModelSelector.currentNode(); tgtOrig = self.targetModelSelector.currentNode()
    corresOrig  = self.sourceFiducialSelector.currentNode(); lmOrig  = self.sourceSparseFiducialSelector.currentNode()
    self.clearCloneFolder(); run_label = f"{srcOrig.GetName()}→{tgtOrig.GetName()}"; self.ensureCloneFolder(label=run_label)
    self.srcTemp = self.cloneNode(srcOrig, customName="Source")
    self.tgtTemp = self.cloneNode(tgtOrig, customName="Target")
    self.corresTemp  = self.cloneNode(corresOrig,  customName="Source Correspondences")
    self.lmTemp  = self.cloneNode(lmOrig,  customName="Landmarks")
    if not  self.parameterDictionary.get("skipScaling", False):
        size_src=bounds_diag(self.srcTemp); size_tgt=bounds_diag(self.tgtTemp); self.scale = (size_tgt/size_src) if size_src>0 else 1.0
        t = vtk.vtkTransform(); t.Scale(self.scale, self.scale, self.scale)
        tn = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode','PreScale'); tn.SetAndObserveTransformToParent(t)
        for node in (self.srcTemp, self.corresTemp, self.lmTemp): node.SetAndObserveTransformNodeID(tn.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(node)
        slicer.mrmlScene.RemoveNode(tn)
    else: self.scale = 1.0
    self.sourceData, self.targetData, self.sourcePoints, self.targetPoints, self.sourceFeatures, self.targetFeatures, self.voxelSize = \
      logic.runSubsample(self.srcTemp, self.tgtTemp, self.parameterDictionary.get("skipScaling", False), self.parameterDictionary)
    if self.sourceData is None:
      slicer.util.errorDisplay("Subsampling failed. Check input files and logs."); return
    src_np = np.asarray(self.sourcePoints.points); tgt_np = np.asarray(self.targetPoints.points)
    self.sourceSLM_vtk = logic.convertPointsToVTK(src_np); self.targetSLM_vtk = logic.convertPointsToVTK(tgt_np)
    self.targetCloudNode = logic.displayPointCloud(self.targetSLM_vtk, self.voxelSize/10, 'Target Pointcloud', COLORS["blue"])
    self._parentNode(self.targetCloudNode); self.updateLayout(); self.alignButton.enabled = True
    self.subsampleInfo.clear(); self.subsampleInfo.insertPlainText(f':: Your subsampled source pointcloud has {len(src_np)} points.\n')
    self.subsampleInfo.insertPlainText(f':: Your subsampled target pointcloud has {len(tgt_np)} points.')

  def onAlignButton(self):
    logic = PREDICTLogic()
    self.transformMatrix = logic.estimateTransform(self.sourcePoints, self.targetPoints, self.sourceFeatures, self.targetFeatures, self.voxelSize, self.parameterDictionary)
    self.ICPTransformNode = logic.convertMatrixToTransformNode(self.transformMatrix, 'Rigid Transformation Matrix')
    self._parentNode(self.ICPTransformNode)
    self.alignedSourceSLM_vtk = logic.applyTransform(self.ICPTransformNode, self.sourceSLM_vtk)
    self.sourceCloudNode = logic.displayPointCloud(self.sourceSLM_vtk, self.voxelSize/10, 'Source Pointcloud', COLORS["red"])
    self.sourceCloudNode.SetAndObserveTransformNodeID(self.ICPTransformNode.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(self.sourceCloudNode)
    self._parentNode(self.sourceCloudNode); self.updateLayout(); self.displayMeshButton.enabled = True; self.alignButton.enabled = False

  def onDisplayMeshButton(self):
    self.tgtTemp.CreateDefaultDisplayNodes(); self.tgtTemp.GetDisplayNode().SetColor(*COLORS["blue"]); self.tgtTemp.GetDisplayNode().SetVisibility(True)
    self.srcTemp.CreateDefaultDisplayNodes(); self.srcTemp.SetAndObserveTransformNodeID(self.ICPTransformNode.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(self.srcTemp)
    self.srcTemp.GetDisplayNode().SetColor(*COLORS["red"]); self.srcTemp.GetDisplayNode().SetVisibility(True)
    self.sourceCloudNode.GetDisplayNode().SetVisibility(False); self.targetCloudNode.GetDisplayNode().SetVisibility(False)
    self.corresTemp.SetAndObserveTransformNodeID(self.ICPTransformNode.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(self.corresTemp); self.corresTemp.GetDisplayNode().SetVisibility(False)
    self.lmTemp.SetAndObserveTransformNodeID(self.ICPTransformNode.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(self.lmTemp); self.lmTemp.GetDisplayNode().SetVisibility(True)
    self.updateLayout(); self.CPDRegistrationButton.enabled = True; self.displayMeshButton.enabled = False

  def onCPDRegistration(self):
    logic = PREDICTLogic()
    alignedSourceCorres_np = slicer.util.arrayFromMarkupsControlPoints(self.corresTemp)
    alignedSourceLM_np = slicer.util.arrayFromMarkupsControlPoints(self.lmTemp)
    tableNode = self.ssmTableSelector.currentNode(); logic.rigidTransformNode = self.ICPTransformNode
    deformedLandmark_np = logic.runDeformable(tableNode, alignedSourceCorres_np, self.scale, self.targetPoints.points, self.parameterDictionary)
    # preview the deformed correspondences as a point cloud
    if getattr(self, "deformedCloudNode", None):
        try: slicer.mrmlScene.RemoveNode(self.deformedCloudNode)
        except: pass
    pc_vtk = logic.convertPointsToVTK(deformedLandmark_np)
    self.deformedCloudNode = logic.displayPointCloud(pc_vtk, self.voxelSize/10, "Warped Source Pointcloud", COLORS["green"]); self.deformedCloudNode.GetDisplayNode().SetVisibility(False)
    self._parentNode(self.deformedCloudNode)
    tree = cKDTree(alignedSourceCorres_np); disp = deformedLandmark_np - alignedSourceCorres_np
    self.warpedMeshNode = self.cloneNode(self.srcTemp, customName="Warped Source"); logic.warp_node(node_to_warp=self.warpedMeshNode, tree=tree, disp=disp)
    logic.smoothModel(self.warpedMeshNode)
    self.warpedLM = self.cloneNode(self.lmTemp, customName="Landmark Predictions"); logic.warp_node(node_to_warp=self.warpedLM, tree=tree, disp=disp)
    self.refinedLM = logic.runPointProjection(self.warpedMeshNode, self.tgtTemp, self.warpedLM, self._proj_frac()); self._parentNode(self.refinedLM)
    self.warpedMeshNode.GetDisplayNode().SetColor(*COLORS["green"]); self.warpedMeshNode.GetDisplayNode().SetVisibility(False)
    self.lmTemp.GetDisplayNode().SetVisibility(False); self.srcTemp.GetDisplayNode().SetVisibility(False); self.warpedLM.GetDisplayNode().SetVisibility(False); self.refinedLM.GetDisplayNode().SetVisibility(True)
    self.displayWarpedModelButton.enabled = True; self.CPDRegistrationButton.enabled = False

  def onDisplayWarpedModel(self):
    self.warpedMeshNode.GetDisplayNode().SetColor(*COLORS["green"]); self.warpedMeshNode.GetDisplayNode().SetVisibility(True)

  def onResetButton(self):
    self.clearCloneFolder(); self._showInputNodes(True)
    for btn in (self.subsampleButton, self.alignButton, self.displayMeshButton, self.CPDRegistrationButton, self.displayWarpedModelButton, self.applyLandmarkMultiButton): btn.enabled=False
    self.onSelect(); slicer.util.showStatusMessage("Reset complete: cleared previous runs", 2000)

  # ----- Batch processing -----
  def onApplyLandmarkMulti(self):
    logic = PREDICTLogic(); projectionFactor = self._proj_frac(); d=self.parameterDictionary
    self._cancelBatch=False; self.batchProgress.setValue(0); self.batchCancelButton.enabled=True
    def progress_cb(done, total, label=None):
      pct = int(100.0 * done / max(1, total)); self.batchProgress.setValue(pct)
      if label: slicer.util.showStatusMessage(f"PREDICT batch: {label}", 1500)
      slicer.app.processEvents()
    def status_cb(label): slicer.util.showStatusMessage(label, 1500); slicer.app.processEvents()
    def cancel_cb(): return self._cancelBatch
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    try:
      logic.runLandmarkBatch(
        sourceModelNode=self.sourceModelMultiSelector.currentNode(),
        sourceCorrNode=self.sourceFiducialMultiSelector.currentNode(),
        sourceLMNode=self.sourceSparseFiducialMultiSelector.currentNode(),
        targetModelDir=self.targetModelMultiSelector.currentPath,
        outputDir=self.landmarkOutputSelector.currentPath,
        skipScaling=d.get("skipScaling", False),
        projectionFactor=projectionFactor,
        parameters=self.parameterDictionary,
        tableNode=self.ssmTableMultiSelector.currentNode(),
        progress_callback=progress_cb,
        status_callback=status_cb,
        cancel_callback=cancel_cb
      )
    finally:
      qt.QApplication.restoreOverrideCursor(); self.batchCancelButton.enabled=False
      if not self._cancelBatch:
        self.batchProgress.setValue(100); slicer.util.showStatusMessage("Batch processing complete.", 3000)
      else:
        slicer.util.showStatusMessage("Batch cancelled.", 3000)

  # ----- Optimization -----
  def onOptimize(self):
    logic = PREDICTLogic()
    tplModelOrig = self.d2sourceModelSelector.currentNode(); tplCorrOrig  = self.d2sourceFiducialSelector.currentNode()
    tplLandOrig  = self.d2sourceSparseFiducialSelector.currentNode(); targetNode   = self.d2targetModelSelector.currentNode(); tableNode = self.d2ssmTableSelector.currentNode()
    if not all([tplModelOrig, tplCorrOrig, targetNode, tableNode]): slicer.util.errorDisplay("Select template model/correspondences, target, and SSM table."); return
    label = f"{tplModelOrig.GetName()}→{targetNode.GetName()}"; self.clearCloneFolder(); self.ensureCloneFolder(label=label)
    tplModel = self.cloneNode(tplModelOrig, "GridRANSAC_TemplateModel"); tplCorr = self.cloneNode(tplCorrOrig,  "GridRANSAC_TemplateCorrespondences")
    tplLand  = self.cloneNode(tplLandOrig,  "GridRANSAC_TemplateLandmarks") if tplLandOrig else None
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    try:
      b, T = logic.initialize_template(
        tableNode, tplModel, tplCorr, tplLand, targetNode,
        parameters=self.parameterDictionary, k=3, span=3.0, optimizer="powell",
        max_evals=120, eval_ransac_iters=int(self.ransacItersPerCand.value * 0.2), final_ransac_iters=int(self.ransacItersPerCand.value), seed=0
      )
    except Exception as e:
      slicer.util.errorDisplay(f"Template optimization failed:\n{e}"); return
    finally:
      qt.QApplication.restoreOverrideCursor()
    txNode = make_transform_node(T, 'SSM gridRANSAC rigid')
    for n in [tplModel, tplCorr] + ([tplLand] if tplLand else []): n.SetAndObserveTransformNodeID(txNode.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(n)
    logic.rigidTransformNode = txNode
    self._parentNode(txNode); self._parentNode(tplModel); self._parentNode(tplCorr); 
    if tplLand: self._parentNode(tplLand)
    sh = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    resRoot = sh.GetItemByName("Template optimization") or sh.CreateFolderItem(sh.GetSceneItemID(), "Template optimization")
    for n in [tplModel, tplCorr] + ([tplLand] if tplLand else []): sh.SetItemParent(sh.GetItemByDataNode(n), resRoot)
    self.cloneFolderItemID = None
    self.sourceModelSelector.setCurrentNode(tplModel); self.sourceFiducialSelector.setCurrentNode(tplCorr)
    if tplLand: self.sourceSparseFiducialSelector.setCurrentNode(tplLand)
    slicer.util.showStatusMessage(f"Template optimized by SSM grid + RANSAC. ||b||={np.linalg.norm(b):.3f}", 4000)

# ---------- Logic ----------
class PREDICTLogic(ScriptedLoadableModuleLogic):
  def __init__(self): super().__init__(); self.rigidTransformNode=None

  def runLandmarkBatch(self, sourceModelNode, sourceCorrNode, sourceLMNode, targetModelDir, outputDir, skipScaling, projectionFactor, parameters, tableNode, progress_callback=None, status_callback=None, cancel_callback=None):
    import tiny3d as t3d
    skipOpt = bool(parameters.get("skipOptimization", False))

    if tableNode is None: raise ValueError("Batch requires an SSM table.")
    if sourceCorrNode is None: raise ValueError("Batch requires template correspondences.")
    if sourceLMNode   is None: raise ValueError("Batch requires template landmarks.")
    os.makedirs(outputDir, exist_ok=True)
    targets=[f for f in os.listdir(targetModelDir) if f.lower().endswith((".ply",".vtp",".vtk",".stl",".obj"))]
    total=len(targets); done=0
    if total==0:
      if status_callback: status_callback("No target meshes found in the selected folder.")
      if progress_callback: progress_callback(0,1,"No work"); return
    app=slicer.app
    try:
      prev_render = app.isRenderPaused()
    except Exception:
      prev_render = False
    app.setRenderPaused(True); scene=slicer.mrmlScene; scene.StartState(slicer.vtkMRMLScene.BatchProcessState)
    def _all_node_ids(): return set(n.GetID() for n in slicer.util.getNodes('*').values())
    def _remove_nodes_by_ids(ids):
      for n in list(slicer.util.getNodes('*').values()):
        try:
          if n.GetID() in ids: scene.RemoveNode(n)
        except: pass
    def _apply_pd_transform(pd, T): return transform_polydata(pd, T)
    def _mat4(M): return as_np4x4(M)
    def _apply_M_to_np(P, M4): P=np.asarray(P, np.float32); M4=_mat4(M4); Ph=np.c_[P, np.ones((len(P),1), np.float32)]; return (Ph @ M4.T)[:, :3]
    tpl_model = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode','_tmp_tpl_model'); tpl_model.SetAttribute('fastmorph.temp','1')
    tpl_corr  = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode','_tmp_tpl_corr'); tpl_corr.SetAttribute('fastmorph.temp','1')
    tpl_lm    = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode','_tmp_tpl_lm'); tpl_lm.SetAttribute('fastmorph.temp','1')
    save_scratch = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode','_tmp_save'); save_scratch.SetAttribute('fastmorph.temp','1')
    for n in (tpl_model, tpl_corr, tpl_lm, save_scratch):
      dn=n.GetDisplayNode()
      if dn:
        dn.SetVisibility(False)
        if hasattr(dn,'SetPointLabelsVisibility'): dn.SetPointLabelsVisibility(False)
    tpl_model_orig=vtk.vtkPolyData(); tpl_model_orig.DeepCopy(sourceModelNode.GetPolyData())
    tpl_corr_orig =slicer.util.arrayFromMarkupsControlPoints(sourceCorrNode).astype(np.float32,copy=True)
    tpl_lm_orig   =slicer.util.arrayFromMarkupsControlPoints(sourceLMNode).astype(np.float32,copy=True)
    try:
      if progress_callback: progress_callback(0, total, "Starting…")
      for i,fname in enumerate(targets):
        baseline_ids = _all_node_ids()
        if cancel_callback and cancel_callback(): logging.info("Batch cancel requested."); break
        tgt_node = icpNode = pred_node = None
        try:
          tgt_path=os.path.join(targetModelDir,fname)
          if status_callback: status_callback(f"[{i+1}/{total}] Load target")
          tgt_node=slicer.util.loadModel(tgt_path)
          if tgt_node is None or tgt_node.GetPolyData() is None: raise RuntimeError(f"Failed to load: {tgt_path}")
          dn=tgt_node.GetDisplayNode(); 
          if dn: dn.SetVisibility(False)
          pd=vtk.vtkPolyData(); pd.DeepCopy(tpl_model_orig); tpl_model.SetAndObservePolyData(pd)
          with slicer.util.NodeModify(tpl_corr): slicer.util.updateMarkupsControlPointsFromArray(tpl_corr, tpl_corr_orig)
          with slicer.util.NodeModify(tpl_lm):   slicer.util.updateMarkupsControlPointsFromArray(tpl_lm,   tpl_lm_orig)
          if not skipOpt:
            if status_callback: status_callback(f"[{i+1}/{total}] Optimize (SSM+RANSAC)…")
            _b,_T=self.initialize_template(tableNode, tpl_model, tpl_corr, tpl_lm, tgt_node, parameters=parameters, k=3, span=3.0, optimizer="powell", max_evals=120, eval_ransac_iters=30000, final_ransac_iters=150000, seed=0)
          else:
            if status_callback: status_callback(f"[{i+1}/{total}] Skip optimization")
          b_src=np.array(tpl_model.GetPolyData().GetBounds()).reshape(3,2); b_tgt=np.array(tgt_node.GetPolyData().GetBounds()).reshape(3,2)
          size_src=np.linalg.norm(b_src[:,1]-b_src[:,0]); size_tgt=np.linalg.norm(b_tgt[:,1]-b_tgt[:,0])
          s=(size_tgt/size_src) if size_src>0 else 1.0
          T_scale=np.eye(4, dtype=np.float32); T_scale[:3,:3]*=s
          tpl_model.SetAndObservePolyData(_apply_pd_transform(tpl_model.GetPolyData(), T_scale))
          Ms=T_scale
          corr_np=_apply_M_to_np(slicer.util.arrayFromMarkupsControlPoints(tpl_corr), Ms)
          lm_np  =_apply_M_to_np(slicer.util.arrayFromMarkupsControlPoints(tpl_lm),   Ms)
          with slicer.util.NodeModify(tpl_corr): slicer.util.updateMarkupsControlPointsFromArray(tpl_corr, corr_np)
          with slicer.util.NodeModify(tpl_lm):   slicer.util.updateMarkupsControlPointsFromArray(tpl_lm,   lm_np)
          if cancel_callback and cancel_callback(): raise KeyboardInterrupt("Cancel requested")
          if status_callback: status_callback(f"[{i+1}/{total}] Subsample & features…")
          _src_pc,_tgt_pc,src_down,tgt_down,src_fpfh,tgt_fpfh,voxel=self.runSubsample(tpl_model, tgt_node, skipScaling, parameters)
          if status_callback: status_callback(f"[{i+1}/{total}] RANSAC+ICP rigid…")
          M_icp=_mat4(self.estimateTransform(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel, parameters))
          tpl_model.SetAndObservePolyData(_apply_pd_transform(tpl_model.GetPolyData(), M_icp))
          corr_np=_apply_M_to_np(corr_np, M_icp); lm_np=_apply_M_to_np(lm_np, M_icp)
          with slicer.util.NodeModify(tpl_corr): slicer.util.updateMarkupsControlPointsFromArray(tpl_corr, corr_np)
          with slicer.util.NodeModify(tpl_lm):   slicer.util.updateMarkupsControlPointsFromArray(tpl_lm,   lm_np)
          icpNode = make_transform_node(M_icp, '_tmp_icp'); icpNode.SetAttribute('fastmorph.temp','1')
          self.rigidTransformNode = icpNode
          if cancel_callback and cancel_callback(): raise KeyboardInterrupt("Cancel requested")
          if status_callback: status_callback(f"[{i+1}/{total}] PCA-CPD deformable…")
          aligned_corr=corr_np
          deformed_corr=self.runDeformable(tableNode=tableNode, sourceLM=aligned_corr, scale=s, targetSLM=np.asarray(tgt_down.points), parameters=parameters)
          if cancel_callback and cancel_callback(): raise KeyboardInterrupt("Cancel requested")
          tree=cKDTree(aligned_corr); disp=(deformed_corr - aligned_corr).astype(np.float32,copy=False)
          self.warp_node(node_to_warp=tpl_model, tree=tree, disp=disp)
          self.warp_node(node_to_warp=tpl_lm,    tree=tree, disp=disp)
          if projectionFactor and projectionFactor>0:
            if status_callback: status_callback(f"[{i+1}/{total}] Project to surface…")
            pred_node=self.runPointProjection(tpl_model, tgt_node, tpl_lm, projectionFactor)
            self.propagateLandmarkTypes(sourceLMNode, pred_node)
            pred_np=slicer.util.arrayFromMarkupsControlPoints(pred_node).astype(np.float32,copy=False)
          else:
            self.propagateLandmarkTypes(sourceLMNode, tpl_lm)
            pred_np=slicer.util.arrayFromMarkupsControlPoints(tpl_lm).astype(np.float32,copy=False)
          root=os.path.splitext(fname)[0]; out_path=os.path.join(outputDir, root+".mrk.json")
          if status_callback: status_callback(f"[{i+1}/{total}] Save landmarks…")
          with slicer.util.NodeModify(save_scratch): slicer.util.updateMarkupsControlPointsFromArray(save_scratch, pred_np)
          slicer.util.saveNode(save_scratch, out_path)
        except KeyboardInterrupt:
          logging.info(f"[batch] Cancelled during {fname}"); raise
        except Exception as e:
          logging.error(f"[batch] Failed on {fname}: {e}")
        finally:
          for n in (pred_node, tgt_node, icpNode):
            if n:
              try: scene.RemoveNode(n)
              except: pass
          self.rigidTransformNode = None
          after_ids = _all_node_ids(); stray_ids = after_ids - baseline_ids
          if stray_ids: _remove_nodes_by_ids(stray_ids)
          done += 1
          if progress_callback: progress_callback(done,total,f"Done {done}/{total}")
      if status_callback and done==total: status_callback("Batch processing complete.")
    except KeyboardInterrupt:
      logging.info(f"[batch] Cancelled at {done}/{total}")
      if progress_callback: progress_callback(done,total,f"Cancelled at {done}/{total}")
      if status_callback: status_callback("Batch cancelled.")
    except Exception as e:
      logging.exception("Batch crashed.")
      if status_callback: status_callback(f"Batch error: {e}")
    finally:
      for n in (tpl_model, tpl_corr, tpl_lm, save_scratch, getattr(self, 'rigidTransformNode', None)):
        if n:
          try: scene.RemoveNode(n)
          except: pass
      scene.EndState(slicer.vtkMRMLScene.BatchProcessState)
      try: app.setRenderPaused(prev_render)
      except:
        try: app.setRenderPaused(False)
        except: pass

  def _tableToArray(self, tableNode):
    vtk_table = tableNode.GetTable(); n_rows = vtk_table.GetNumberOfRows(); n_cols = vtk_table.GetNumberOfColumns()
    arr = np.zeros((n_rows, n_cols), dtype=float)
    for j in range(n_cols): name = vtk_table.GetColumn(j).GetName(); arr[:, j] = slicer.util.arrayFromTableColumn(tableNode, name)
    return arr

  def warp_node(self, node_to_warp, tree, disp, k=16):
    if isinstance(node_to_warp, slicer.vtkMRMLModelNode):
      original_points = slicer.util.arrayFromModelPoints(node_to_warp)
    elif isinstance(node_to_warp, slicer.vtkMRMLMarkupsFiducialNode):
      original_points = slicer.util.arrayFromMarkupsControlPoints(node_to_warp)
    else:
      logging.error(f"Unsupported node type for in-place warping: {type(node_to_warp)}"); return
    interpolated_disp = self.interpolate_displacements(original_points, tree, disp, k=k)
    new_points = original_points + interpolated_disp
    if isinstance(node_to_warp, slicer.vtkMRMLModelNode):
      points = node_to_warp.GetPolyData().GetPoints(); vtk_array = points.GetData(); vtk_type = vtk_array.GetDataType() if vtk_array else vtk.VTK_DOUBLE
      vtk_array_new = vtk_np.numpy_to_vtk(new_points, deep=True, array_type=vtk_type); points.SetData(vtk_array_new); points.Modified(); node_to_warp.GetPolyData().Modified()
    else:
      with slicer.util.NodeModify(node_to_warp): slicer.util.updateMarkupsControlPointsFromArray(node_to_warp, new_points)

  def interpolate_displacements(self, verts, tree, disp, k=12, sigma=None):
    d, idx = tree.query(verts, k=k)
    if sigma is None: sigma = np.median(d[:, -1]) + 1e-12
    w = np.exp(-(d**2) / (2 * sigma**2)); w /= (w.sum(1, keepdims=True) + 1e-12)
    return (disp[idx] * w[..., None]).sum(1)

  def smoothModel(self, modelNode, iterations=10, passBand=0.08):
    if not modelNode or not modelNode.GetPolyData(): logging.error("Invalid model node for smoothing."); return
    f = vtk.vtkWindowedSincPolyDataFilter(); f.SetInputData(modelNode.GetPolyData())
    f.SetNumberOfIterations(iterations); f.SetPassBand(passBand); f.NormalizeCoordinatesOff(); f.Update()
    modelNode.SetAndObservePolyData(f.GetOutput())

  def runDeformable(self, tableNode, sourceLM, scale, targetSLM, parameters):
      from biocpd.atlas_registration import AtlasRegistration
      if tableNode is None: raise ValueError("runDeformable requires tableNode to be set")

      # Keep SSM geometry consistent with the (already scaled) sourceLM
      flat = self._tableToArray(tableNode) * float(scale)
      M = flat.shape[0] // 3
      if M != sourceLM.shape[0]:
          raise ValueError(f"SSM table has {M} points but sourceLM has {sourceLM.shape[0]}")

      modes = flat[:, 1:].reshape(M, 3, -1)

      # Eigenvalues: unscaled; drop numerically tiny modes (avoids jitter)
      eig = np.array(json.loads(tableNode.GetAttribute("ssm_eigenvalues")), float)
      rel = eig / (eig.max() + 1e-12)
      keep = (eig > 0) & (rel > 1e-8)
      if not np.all(keep):
          modes = modes[:, :, keep]
          eig = eig[keep]
      eigvals_eff = eig

      # Rigid rotation for the modes
      if self.rigidTransformNode is None:
          raise ValueError("runDeformable requires that logic.rigidTransformNode be set")
      mat = vtk.vtkMatrix4x4(); self.rigidTransformNode.GetMatrixTransformToParent(mat)
      T = vtk_mat_to_np(mat); R = T[:3, :3]

      U_aligned = np.einsum('ij,pjk->pik', R, modes).reshape(3*M, -1)

      pca = AtlasRegistration(
          X=np.asarray(targetSLM),
          Y=np.asarray(sourceLM),            # mean_shape=None ⇒ Y is the base
          mean_shape=None,
          U=U_aligned,
          eigenvalues=eigvals_eff,          # no external scaling/flooring
          lambda_reg=float(parameters.get("lambda_reg", 0.4)),  # no scale² here
          alpha=float(parameters.get("alpha", 2.0)),
          w=float(parameters.get("w", 0.1)),
          tolerance=float(parameters.get("tolerance", 1e-6)),
          max_iterations=int(parameters.get("max_iterations", 120)),
          normalize=True                    # << key change
      )
      warped_landmarks, _ = pca.register()
      return warped_landmarks


  def convertMatrixToTransformNode(self, matrix, transformName):
    return make_transform_node(matrix, transformName)

  def applyTransform(self, transform_input, polydata):
    if isinstance(transform_input, slicer.vtkMRMLTransformNode):
      mat = vtk.vtkMatrix4x4(); transform_input.GetMatrixTransformToParent(mat); M = vtk_mat_to_np(mat)
    elif isinstance(transform_input, vtk.vtkMatrix4x4):
      M = vtk_mat_to_np(transform_input)
    else:
      raise TypeError("applyTransform expects a vtkMRMLTransformNode or vtkMatrix4x4")
    return transform_polydata(polydata, M)

  def convertPointsToVTK(self, points):
    array_vtk = vtk_np.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT)
    points_vtk = vtk.vtkPoints(); points_vtk.SetData(array_vtk)
    polydata_vtk = vtk.vtkPolyData(); polydata_vtk.SetPoints(points_vtk)
    return polydata_vtk

  def displayPointCloud(self, polydata, pointRadius, nodeName, nodeColor):
    vgf = vtk.vtkVertexGlyphFilter(); vgf.SetInputData(polydata); vgf.Update()
    node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', nodeName); node.CreateDefaultDisplayNodes()
    node.SetAndObservePolyData(vgf.GetOutput()); node.GetDisplayNode().SetColor(nodeColor)
    nd = node.GetDisplayNode()
    if hasattr(nd,'SetPointSize'): nd.SetPointSize(max(1,int(round(pointRadius*100))))  # visual size
    return node


  def estimateTransform(self, sourcePoints, targetPoints, sourceFeatures, targetFeatures, voxelSize, parameters):
    import tiny3d as t3d
    distanceThreshold = voxelSize * parameters["distanceThreshold"]
    if not sourcePoints.has_normals():
      sourcePoints.estimate_normals(t3d.geometry.KDTreeSearchParamHybrid(radius=voxelSize * parameters["normalSearchRadius"], max_nn=30))
    if not targetPoints.has_normals():
      targetPoints.estimate_normals(t3d.geometry.KDTreeSearchParamHybrid(radius=voxelSize * parameters["normalSearchRadius"], max_nn=30))
    no_scaling = t3d.pipelines.registration.registration_ransac_based_on_feature_matching(
      sourcePoints, targetPoints, sourceFeatures, targetFeatures, True, distanceThreshold,
      t3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
      [t3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), t3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distanceThreshold)],
      t3d.pipelines.registration.RANSACConvergenceCriteria(parameters["maxRANSAC"], confidence=parameters["confidence"]))
    no_scaling_eval = t3d.pipelines.registration.evaluate_registration(targetPoints, sourcePoints, distanceThreshold, np.linalg.inv(no_scaling.transformation))
    best_result=no_scaling; fitness=(no_scaling.fitness + no_scaling_eval.fitness)/2
    count=0; maxAttempts=4
    try:
      while fitness < 0.99 and count < maxAttempts:
        result = t3d.pipelines.registration.registration_ransac_based_on_feature_matching(
          sourcePoints, targetPoints, sourceFeatures, targetFeatures, True, distanceThreshold,
          t3d.pipelines.registration.TransformationEstimationPointToPoint(True), 3,
          [t3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), t3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distanceThreshold)],
          t3d.pipelines.registration.RANSACConvergenceCriteria(parameters["maxRANSAC"], confidence=parameters["confidence"]))
        evaluation = t3d.pipelines.registration.evaluate_registration(targetPoints, sourcePoints, distanceThreshold, np.linalg.inv(result.transformation))
        mean_fitness = (result.fitness + evaluation.fitness)/2
        if mean_fitness > fitness: fitness = mean_fitness; best_result = result
        count += 1
    except: pass
    icp = t3d.pipelines.registration.registration_icp(
      sourcePoints, targetPoints, voxelSize * parameters["ICPDistanceThreshold"], best_result.transformation,
      t3d.pipelines.registration.TransformationEstimationPointToPlane())
    return icp.transformation

  def runSubsample(self, sourceNode, targetNode, skipScaling, parameters):
    import tiny3d as t3d
    src_pd = sourceNode.GetPolyData(); tgt_pd = targetNode.GetPolyData()
    sourcePoints_np = vtk_np.vtk_to_numpy(src_pd.GetPoints().GetData()); targetPoints_np = vtk_np.vtk_to_numpy(tgt_pd.GetPoints().GetData())
    source = t3d.geometry.PointCloud(); source.points = t3d.utility.Vector3dVector(sourcePoints_np)
    target = t3d.geometry.PointCloud(); target.points = t3d.utility.Vector3dVector(targetPoints_np)
    if skipScaling:
      size = np.linalg.norm(target.get_max_bound() - target.get_min_bound()); voxel_size = size / (55 * parameters["pointDensity"]) ; source_center = np.zeros(3); target_center = np.zeros(3); source_scaling = target_scaling = 1.0
    else:
      voxel_size = 1.0 / (55 * parameters["pointDensity"]) ; source_center = source.get_center(); target_center = target.get_center()
      tmp_src = copy.deepcopy(source).translate(-source_center); tmp_tgt = copy.deepcopy(target).translate(-target_center)
      sourceSize = np.linalg.norm(tmp_src.get_max_bound() - tmp_src.get_min_bound()); targetSize = np.linalg.norm(tmp_tgt.get_max_bound() - tmp_tgt.get_min_bound())
      source_scaling = 1.0 / sourceSize if sourceSize > 0 else 1.0; target_scaling = 1.0 / targetSize if targetSize > 0 else 1.0
      source.scale(source_scaling, center=source_center); target.scale(target_scaling, center=target_center)
    source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size, parameters["normalSearchRadius"], parameters["FPFHSearchRadius"]) 
    target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size, parameters["normalSearchRadius"], parameters["FPFHSearchRadius"]) 
    if not skipScaling:
      src_pts = np.asarray(source_down.points); tgt_pts = np.asarray(target_down.points)
      src_pts = (src_pts - source_center) / source_scaling + source_center
      tgt_pts = (tgt_pts - target_center) / target_scaling + target_center
      source_down = t3d.geometry.PointCloud(); source_down.points = t3d.utility.Vector3dVector(src_pts)
      target_down = t3d.geometry.PointCloud(); target_down.points = t3d.utility.Vector3dVector(tgt_pts)
      voxel_size = voxel_size / target_scaling
    return (source, target, source_down, target_down, source_fpfh, target_fpfh, voxel_size)

  def propagateLandmarkTypes(self, sourceNode, targetNode):
    if sourceNode.GetNumberOfControlPoints() != targetNode.GetNumberOfControlPoints(): logging.warning("Source and target nodes have different number of landmarks"); return
    for i in range(sourceNode.GetNumberOfControlPoints()):
      if hasattr(sourceNode, 'GetNthControlPointDescription'):
        targetNode.SetNthControlPointDescription(i, sourceNode.GetNthControlPointDescription(i))
      if hasattr(sourceNode, 'GetNthControlPointLabel'):
        targetNode.SetNthControlPointLabel(i, sourceNode.GetNthControlPointLabel(i))

  def preprocess_point_cloud(self, pcd, voxel_size, radius_normal_factor, radius_feature_factor):
    import tiny3d as t3d
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) == 0:
      raise RuntimeError(f"Downsampling produced 0 points at voxel={voxel_size:.4f}. Increase point density or reduce voxel size.")
    radius_normal = voxel_size * radius_normal_factor
    pcd_down.estimate_normals(t3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * radius_feature_factor
    pcd_fpfh = t3d.pipelines.registration.compute_fpfh_feature(pcd_down, t3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

  def runPointProjection(self, template, model, templateLandmarks, maxProjectionFactor):
    maxProjection = (model.GetPolyData().GetLength()) * maxProjectionFactor
    templatePoints = self.getFiducialPoints(templateLandmarks)
    normalFilter = vtk.vtkPolyDataNormals(); normalFilter.SetInputData(template.GetPolyData()); normalFilter.ComputePointNormalsOn(); normalFilter.SplittingOff(); normalFilter.Update()
    sourcePolydata = normalFilter.GetOutput()
    projectedPoints = self.projectPointsPolydata(sourcePolydata, model.GetPolyData(), templatePoints, maxProjection)
    projectedLMNode= slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',"Refined Predicted Landmarks")
    with slicer.util.NodeModify(projectedLMNode):
      for i in range(projectedPoints.GetNumberOfPoints()): projectedLMNode.AddControlPoint(projectedPoints.GetPoint(i))
      projectedLMNode.SetLocked(True); projectedLMNode.SetFixedNumberOfControlPoints(True)
    return projectedLMNode

  def getFiducialPoints(self,fiducialNode):
    points = vtk.vtkPoints()
    for i in range(fiducialNode.GetNumberOfControlPoints()): points.InsertNextPoint(fiducialNode.GetNthControlPointPosition(i))
    return points

  def projectPointsPolydata(self, sourcePolydata, targetPolydata, originalPoints, rayLength):
    sourcePointLocator = vtk.vtkStaticPointLocator(); sourcePointLocator.SetDataSet(sourcePolydata); sourcePointLocator.BuildLocator()
    targetPointLocator = vtk.vtkStaticPointLocator(); targetPointLocator.SetDataSet(targetPolydata); targetPointLocator.BuildLocator()
    obbTree = vtk.vtkOBBTree(); obbTree.SetDataSet(targetPolydata); obbTree.BuildLocator()
    normalArray = sourcePolydata.GetPointData().GetNormals()
    if not normalArray: return vtk.vtkPolyData()
    projectedPoints = vtk.vtkPoints()
    for i in range(originalPoints.GetNumberOfPoints()):
      p0 = np.array(originalPoints.GetPoint(i)); pid = sourcePointLocator.FindClosestPoint(p0)
      n = np.array(normalArray.GetTuple(pid), float); ln = np.linalg.norm(n)
      if ln < 1e-12:
        tid = targetPointLocator.FindClosestPoint(p0); projectedPoints.InsertNextPoint(targetPolydata.GetPoint(tid)); continue
      n /= ln; candidates = []
      for sgn in (1.0, -1.0):
        p1 = (p0 + sgn*n*rayLength).tolist(); pts = vtk.vtkPoints(); obbTree.IntersectWithLine(p0.tolist(), p1, pts, None)
        for j in range(pts.GetNumberOfPoints()):
          q = np.array(pts.GetPoint(j)); v = q - p0; d = np.linalg.norm(v)
          if d <= 1e-12: continue
          cos = float(np.dot(v/d, n)); candidates.append((cos, d, q))
      if candidates:
        candidates.sort(key=lambda t: (-t[0], t[1])); projectedPoints.InsertNextPoint(candidates[0][2])
      else:
        tid = targetPointLocator.FindClosestPoint(p0); projectedPoints.InsertNextPoint(targetPolydata.GetPoint(tid))
    projectedPointData = vtk.vtkPolyData(); projectedPointData.SetPoints(projectedPoints); return projectedPointData

  def _ssm_unpack(self, tableNode, drop_modes=None):
    A   = self._tableToArray(tableNode)
    mean= A[:,0]; U = A[:,1:]; M = mean.size//3; mean= mean.reshape(M,3); U = U.reshape(M,3,-1)
    eig = np.array(json.loads(tableNode.GetAttribute("ssm_eigenvalues")), float)
    eig[eig<1e-10]=1e-10
    if drop_modes:
      keep = [i for i in range(U.shape[2]) if i not in set(drop_modes)]; U = U[:,:,keep]; eig = eig[keep]
    return mean, U, eig

  def initialize_template(self, tableNode, srcModelNode, srcCorrNode, srcLmNode, tgtModelNode, parameters,
                        k=8, span=2.0, optimizer="powell", max_evals=120,
                        eval_ransac_iters=30000, final_ransac_iters=150000, seed=0):
    import numpy as np, tiny3d as t3d
    # ---- SSM unpack + fast sampler ----
    mean, U, eig = self._ssm_unpack(tableNode)
    k = int(min(k, U.shape[2])); k_eff = max(1, k)
    eig_k = eig[:k]; sqrt_eig_k = np.sqrt(eig_k)
    M = mean.shape[0]
    Uw_flat = (U[:, :, :k].reshape(-1, k) * sqrt_eig_k[None, :])  # (3M × k)
    mean_flat = mean.reshape(-1)

    def ssm_sample(b):  # (k,) -> (M,3)
        return (mean_flat + Uw_flat @ np.asarray(b, float)).reshape(M, 3)

    # ---- Frame: scale target to SSM size (only for evaluation) ----
    def bbox_diag_np(P): P = np.asarray(P); return float(np.linalg.norm(P.max(0) - P.min(0)))
    oldCorr = slicer.util.arrayFromMarkupsControlPoints(srcCorrNode)
    tgt_np = slicer.util.arrayFromModelPoints(tgtModelNode)
    tgt_pcd = t3d.geometry.PointCloud(); tgt_pcd.points = t3d.utility.Vector3dVector(tgt_np)
    s0 = bbox_diag_np(mean) / (bbox_diag_np(tgt_np) + 1e-12)
    tgt_scaled = t3d.geometry.PointCloud(tgt_pcd); tgt_scaled.scale(s0, center=tgt_scaled.get_center())

    # ---- Geometry: coarse→fine targets ----
    size_scaled = float(np.linalg.norm(tgt_scaled.get_max_bound() - tgt_scaled.get_min_bound()))
    voxel_f = size_scaled / (25.0 * float(parameters.get("pointDensity", 1.0)))
    voxel_c = voxel_f * 3.25  # slightly coarser than before for speed
    rn = int(parameters.get("normalSearchRadius", 2))
    rf = int(parameters.get("FPFHSearchRadius", 5))
    tgt_down_f, tgt_fpfh_f = self.preprocess_point_cloud(tgt_scaled, voxel_f, rn, rf)
    tgt_down_c, tgt_fpfh_c = self.preprocess_point_cloud(tgt_scaled, voxel_c, max(1, rn // 2), max(2, rf // 2))
    dist_f = voxel_f * float(parameters.get("distanceThreshold", 3.0))
    dist_c = voxel_c * float(parameters.get("distanceThreshold", 3.0))

    # ---- Candidate set: Sobol + axis seeds ----
    try:
        from scipy.stats import qmc
        N = int(parameters.get("init_candidates", 192)); N = max(96, N)
        dpow = int(np.ceil(np.log2(max(2, N))))
        Sob = qmc.scale(qmc.Sobol(d=k, scramble=True, seed=seed).random_base2(dpow)[:N], -span, span)
    except Exception:
        rng = np.random.default_rng(seed); Sob = rng.uniform(-span, span, size=(max(96, int(parameters.get("init_candidates", 192))), k))
    m = min(k, 10)
    axis = []
    for a in (1.0, 0.5):
        A = np.zeros((m, k)); A[np.arange(m), np.arange(m)] =  a * span; axis.append(A)
        A = np.zeros((m, k)); A[np.arange(m), np.arange(m)] = -a * span; axis.append(A)
    cand = np.unique(np.round(np.vstack([Sob, *axis]), 6), axis=0)

    # ---- Objective weights (consistent prior) ----
    rho = float(parameters.get("reg_strength", 0.15))
    w_rmse = float(parameters.get("w_rmse", 0.30))
    w_reg = min(rho / k_eff, 0.25 / (k_eff * (span**2) + 1e-12))  # keeps ub sane at |b|=span
    LARGE = 1e6; bound_margin = float(parameters.get("bound_margin", 1e-4))

    def clip_b(b): return np.clip(np.asarray(b, float), -span, span)

    # ---- Feature builders (cached) ----
    cand_pcd = t3d.geometry.PointCloud()
    sp_norm_c = t3d.geometry.KDTreeSearchParamHybrid(radius=voxel_c * max(1, int(parameters.get("subsetNormalRadius", rn)) // 2), max_nn=20)
    sp_feat_c = t3d.geometry.KDTreeSearchParamHybrid(radius=voxel_c * max(2, int(parameters.get("subsetFPFHRadius", rf)) // 2), max_nn=60)
    sp_norm_f = t3d.geometry.KDTreeSearchParamHybrid(radius=voxel_f * float(parameters.get("subsetNormalRadius", rn)), max_nn=30)
    sp_feat_f = t3d.geometry.KDTreeSearchParamHybrid(radius=voxel_f * float(parameters.get("subsetFPFHRadius", rf)), max_nn=100)

    max_pts_cand_coarse = int(parameters.get("maxFeatPtsCoarse", 1800))  # cap candidate points for coarse FPFH
    rng = np.random.default_rng(seed)
    feat_cache = {}  # {(b_tuple, coarse): (cand_down, cand_fpfh, td, tf, dth)}

    def get_feat(b, coarse):
        key = (tuple(np.round(np.asarray(b, float), 6)), bool(coarse))
        if key in feat_cache: return feat_cache[key]
        pts = ssm_sample(b)
        if coarse and (max_pts_cand_coarse > 0) and (len(pts) > max_pts_cand_coarse):
            idx = rng.choice(len(pts), size=max_pts_cand_coarse, replace=False)
            pts = pts[idx]
        cand_pcd.points = t3d.utility.Vector3dVector(pts)
        if coarse:
            cand_pcd.estimate_normals(sp_norm_c); fpfh = t3d.pipelines.registration.compute_fpfh_feature(cand_pcd, sp_feat_c)
            pack = (cand_pcd, fpfh, tgt_down_c, tgt_fpfh_c, dist_c)
        else:
            cand_pcd.estimate_normals(sp_norm_f); fpfh = t3d.pipelines.registration.compute_fpfh_feature(cand_pcd, sp_feat_f)
            pack = (cand_pcd, fpfh, tgt_down_f, tgt_fpfh_f, dist_f)
        feat_cache[key] = pack; return pack

    # ---- Scoring (with cache + early bound) ----
    best = {"score": -np.inf, "b": None, "fit": None, "rmse": None, "T": None}
    cache = {}  # {(b_tuple, coarse, iters): neg}

    def eval_one(b, iters, coarse):
        cand_down, cand_fpfh, td, tf, dth = get_feat(b, coarse)
        r = t3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            cand_down, td, cand_fpfh, tf, True, dth,
            t3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [t3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             t3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dth)],
            t3d.pipelines.registration.RANSACConvergenceCriteria(int(iters), float(parameters.get("confidence", 0.9))))
        ev = t3d.pipelines.registration.evaluate_registration(cand_down, td, dth, r.transformation)
        return r.transformation, float(ev.fitness), float(ev.inlier_rmse) / (dth + 1e-12)

    def score_geom(b, iters, coarse):
        b = clip_b(b)
        key = (tuple(np.round(b, 6)), bool(coarse), int(iters))
        if key in cache: return cache[key]["neg"]
        reg = float(np.dot(b, b))
        ub = 1.0 - w_reg * reg
        if best["score"] > -np.inf and ub <= best["score"] - bound_margin:
            cache[key] = {"neg": LARGE + reg}; return cache[key]["neg"]
        if ssm_sample(b).shape != oldCorr.shape:
            cache[key] = {"neg": LARGE}; return cache[key]["neg"]
        try:
            T, fit, rmse = eval_one(b, iters, coarse)
        except Exception:
            cache[key] = {"neg": LARGE}; return cache[key]["neg"]
        score = fit - w_rmse * rmse - w_reg * reg
        if np.isfinite(score) and score > best["score"]:
            best.update(score=score, b=b.copy(), fit=float(fit), rmse=float(rmse), T=T.copy())
        cache[key] = {"neg": -score}; return cache[key]["neg"]

    # ---- Bandit schedule (smaller budgets, fewer keeps) ----
    budgets = [0.005, 0.02, 0.10]  # fractions of eval_ransac_iters
    keeps   = [96,    24,   6]
    C = cand.tolist()
    it = max(1, int(budgets[0] * eval_ransac_iters)); C = sorted(C, key=lambda x: score_geom(x, it, True ))[:min(keeps[0], len(C))]
    it = max(1, int(budgets[1] * eval_ransac_iters)); C = sorted(C, key=lambda x: score_geom(x, it, True ))[:min(keeps[1], len(C))]
    it = max(1, int(budgets[2] * eval_ransac_iters)); C = sorted(C, key=lambda x: score_geom(x, it, False))[:min(keeps[2], len(C))]

    # ---- Local refine: only best finalist, half budget ----
    x0 = np.zeros(k, float)
    if optimizer.lower() == "de":
        from scipy.optimize import differential_evolution
        bounds = [(-span, span)] * k
        res = differential_evolution(lambda x: score_geom(x, eval_ransac_iters, False), bounds=bounds,
                                     strategy="best1bin", popsize=8, maxiter=max(1, max_evals // 8),
                                     tol=1e-3, polish=False, seed=seed, updating="deferred", workers=1)
        b_star = clip_b(res.x)
    else:
        from scipy.optimize import minimize
        start = C[0] if len(C) else x0
        per = max(1, int(0.5 * max_evals))
        r = minimize(lambda x: score_geom(x, eval_ransac_iters, False), x0=clip_b(start),
                     method="Powell", options={"maxfev": per, "xtol": 1e-3, "ftol": 1e-3, "disp": False})
        b_star = clip_b(r.x if (hasattr(r, "x") and r.x is not None) else start)

    _ = score_geom(b_star, final_ransac_iters, False)
    if best["b"] is None: raise RuntimeError("Optimization failed to find a valid candidate.")

    # ---- Commit best shape, no rigid applied here ----
    b_best, T_best = best["b"], best["T"]
    newCorr = ssm_sample(b_best); disp = newCorr - oldCorr; warp_tree = cKDTree(oldCorr)
    srcModelNode.CreateDefaultDisplayNodes(); self.warp_node(node_to_warp=srcModelNode, tree=warp_tree, disp=disp)
    if srcLmNode is not None: self.warp_node(node_to_warp=srcLmNode, tree=warp_tree, disp=disp)
    slicer.util.updateMarkupsControlPointsFromArray(srcCorrNode, newCorr)
    print(f"[opt] ||b||={np.linalg.norm(b_best):.3f} fit={best['fit']:.3f} rmse={best['rmse']:.3f} evals≈{len(cache)}")
    return b_best, T_best


class PREDICTTest(ScriptedLoadableModuleTest):
  def setUp(self): slicer.mrmlScene.Clear(0)
  def runTest(self): self.setUp(); self.test_PREDICT1()
  def test_PREDICT1(self):
    self.delayDisplay("Starting the test"); logic = PREDICTLogic(); self.assertIsNotNone(logic); self.delayDisplay('Test passed!')
