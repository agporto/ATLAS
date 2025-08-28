import os
from datetime import datetime
from pathlib import Path

import numpy as np
import vtk, qt, ctk, slicer
import vtk.util.numpy_support as vtk_np
from slicer.ScriptedLoadableModule import *
from scipy.spatial import cKDTree

class BUILDER(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "BUILDER"
    self.parent.categories = ["ATLAS"]
    self.parent.dependencies = []
    self.parent.contributors = ["Arthur Porto"]
    self.parent.helpText = "Align meshes/landmarks to an atlas and export dense correspondences as .mrk.json"
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = ""

class BUILDERWidget(ScriptedLoadableModuleWidget):
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    root=ctk.ctkCollapsibleButton(); root.text="Build Atlas, Align Specimens, Export Landmarks"; root.collapsed=False
    lay=qt.QFormLayout(root); self.layout.addWidget(root)

    self.createAtlasRadio=qt.QRadioButton("Create atlas from inputs"); self.loadAtlasRadio=qt.QRadioButton("Load existing atlas"); self.createAtlasRadio.setChecked(True)
    g=qt.QButtonGroup(root); g.addButton(self.createAtlasRadio); g.addButton(self.loadAtlasRadio)
    lay.addRow(self.createAtlasRadio); lay.addRow(self.loadAtlasRadio)

    self.atlasBox=ctk.ctkCollapsibleButton(); self.atlasBox.text="Atlas (when loading)"; self.atlasBox.collapsed=True; self.atlasBox.enabled=False
    aLay=qt.QFormLayout(self.atlasBox); lay.addRow(self.atlasBox)
    self.atlasModelPath=ctk.ctkPathLineEdit(); self.atlasModelPath.filters=ctk.ctkPathLineEdit().Files; self.atlasModelPath.nameFilters=["Model (*.ply *.stl *.obj *.vtk *.vtp)"]
    self.atlasLMPath=ctk.ctkPathLineEdit(); self.atlasLMPath.filters=ctk.ctkPathLineEdit().Files; self.atlasLMPath.nameFilters=["Point set (*.fcsv *.mrk.json)"]
    aLay.addRow("Atlas model:", self.atlasModelPath); aLay.addRow("Atlas landmarks:", self.atlasLMPath)

    self.modelDir=ctk.ctkPathLineEdit(); self.modelDir.filters=ctk.ctkPathLineEdit.Dirs; lay.addRow("Model directory:", self.modelDir)
    self.lmDir=ctk.ctkPathLineEdit(); self.lmDir.filters=ctk.ctkPathLineEdit.Dirs; lay.addRow("Landmark directory:", self.lmDir)
    self.outDir=ctk.ctkPathLineEdit(); self.outDir.filters=ctk.ctkPathLineEdit.Dirs; lay.addRow("Output directory:", self.outDir)

    self.useSimilarity = qt.QCheckBox("Normalize scale")
    self.useSimilarity.setChecked(True)
    self.useSimilarity.setToolTip("Checked (recommended): allow isotropic scaling so size differences are removed. Unchecked: rigid only.")
    lay.addRow(self.useSimilarity)

    self.spacing=ctk.ctkSliderWidget(); self.spacing.singleStep=.1; self.spacing.minimum=0; self.spacing.maximum=10; self.spacing.value=4
    self.spacing.setToolTip("Larger radius → fewer exported points")
    lay.addRow("Sampling radius (% of diag):", self.spacing)
    
    self.previewBtn = qt.QPushButton("Preview count"); self.previewBtn.enabled=False
    self.previewLbl = qt.QLabel("")
    h = qt.QHBoxLayout(); h.addWidget(self.previewBtn); h.addWidget(self.previewLbl, 1)
    lay.addRow("Expected points:", h)

    self._previewTimer = qt.QTimer(); self._previewTimer.setSingleShot(True); self._previewTimer.setInterval(250)
    self._previewTimer.timeout.connect(self._onPreview)
    self.spacing.valueChanged.connect(lambda v: self._previewTimer.start())

    self.runBtn=qt.QPushButton("Run BUILDER"); self.runBtn.enabled=False; lay.addRow(self.runBtn)
    self.log=qt.QPlainTextEdit(); self.log.setReadOnly(True); lay.addRow(self.log)

    self.createAtlasRadio.toggled.connect(self._toggleAtlasLoad)
    self.loadAtlasRadio.toggled.connect(self._toggleAtlasLoad)
    self.previewBtn.clicked.connect(self._onPreview)

    for w in [self.atlasModelPath,self.atlasLMPath,self.modelDir,self.lmDir,self.outDir]:
      w.connect('validInputChanged(bool)', self._reeval)
    self.runBtn.clicked.connect(self._onRun)
    self._reeval()

  def _toggleAtlasLoad(self):
      useLoad = self.loadAtlasRadio.isChecked()
      self.atlasBox.collapsed = not useLoad
      self.atlasBox.enabled = useLoad
      self._invalidatePreviewCache()
      self._reeval()
    
  def _reeval(self, *args):
      # Enable/disable main run
      atlasOK = self.createAtlasRadio.isChecked() or bool(self.atlasModelPath.currentPath and self.atlasLMPath.currentPath)
      ioOK    = bool(self.modelDir.currentPath and self.lmDir.currentPath and self.outDir.currentPath)
      self.runBtn.enabled = bool(atlasOK and ioOK)

      # Enable/disable preview button
      hasRef = (self.loadAtlasRadio.isChecked() and bool(self.atlasModelPath.currentPath)) or bool(self.modelDir.currentPath)
      self.previewBtn.enabled = hasRef

      self._invalidatePreviewCache()

  def _loadAtlas(self):
    try:
      m=slicer.util.loadModel(self.atlasModelPath.currentPath); l=slicer.util.loadMarkups(self.atlasLMPath.currentPath); return m,l
    except: self.log.appendPlainText("Failed to load atlas model/landmarks."); return None,None

  def _outFolders(self, base):
    ts = datetime.now().strftime('%Y_%m-%d_%H_%M_%S')
    root = os.path.join(base, ts); os.makedirs(root, exist_ok=True)
    d = {'output': root}
    for k in ['alignedModels','alignedLMs','atlas','dense']:
      p = os.path.join(root, k); os.makedirs(p, exist_ok=True); d[k] = p
    return d

  def _onPreview(self):
      pd = self._getPreviewPolyData()
      if not pd:
          self.previewLbl.setText("n/a")
          return
      n, tot = BUILDERLogic().previewCountForRadius(pd, self.spacing.value)
      pct = (100.0 * n / max(1, tot))
      self.previewLbl.setText(f"{n} of {tot} (~{pct:.1f}%)")

  def _invalidatePreviewCache(self, *_):
      """Clear cached preview PolyData so the next preview recomputes it."""
      self._previewPD = None
      if hasattr(self, "previewLbl"):
          self.previewLbl.setText("")


  def _getPreviewPolyData(self):
      """Return (and cache) a vtkPolyData for preview sampling.

      Prefers:
        1) Loaded atlas model (when 'Load existing atlas' is selected)
        2) First model found in the Models directory
      """
      if getattr(self, "_previewPD", None) is not None:
          return self._previewPD

      try:
          if self.loadAtlasRadio.isChecked() and self.atlasModelPath.currentPath:
              n = slicer.util.loadModel(self.atlasModelPath.currentPath)
              pd = vtk.vtkPolyData(); pd.DeepCopy(n.GetPolyData())
              slicer.mrmlScene.RemoveNode(n)
              self._previewPD = pd
              return pd

          d = self.modelDir.currentPath
          if d:
              for f in sorted(os.listdir(d)):
                  if f.lower().endswith(('.ply', '.stl', '.vtp', '.vtk', '.obj')):
                      n = slicer.util.loadModel(os.path.join(d, f))
                      pd = vtk.vtkPolyData(); pd.DeepCopy(n.GetPolyData())
                      slicer.mrmlScene.RemoveNode(n)
                      self._previewPD = pd
                      return pd
      except Exception:
          pass

      self._previewPD = None
      return None

  def _onRun(self):
    logic = BUILDERLogic()
    F = self._outFolders(self.outDir.currentPath)
    F['originalModels'] = self.modelDir.currentPath
    F['originalLMs']    = self.lmDir.currentPath
    useSimilarity = self.useSimilarity.isChecked()

    # 1) Get atlas (load or build)
    if self.loadAtlasRadio.isChecked():
      atlasModel, atlasLMs = self._loadAtlas()
      if not atlasModel or not atlasLMs: return
    else:
      try:
        atlasModel, atlasLMs = self._generateAtlas(F, useSimilarity)
      except Exception as e:
        self.log.appendPlainText(f"Atlas generation failed: {e}"); return

    # 2) Align all to atlas
    try:
      logic.runAlign(atlasModel, atlasLMs,
                    F['originalModels'], F['originalLMs'],
                    F['alignedModels'],  F['alignedLMs'],
                    useSimilarity,
                    progress=lambda i,n,s: self.log.appendPlainText(f"[{i}/{n}] {s}"))
      
    except ValueError as e:
      self.log.appendPlainText(str(e)); return

    logic.saveAtlasOnly(atlasModel, atlasLMs, F['atlas'])
    logic.exportDenseLMs(atlasModel, atlasLMs,
                        F['alignedModels'], F['alignedLMs'],
                        F['dense'], F['atlas'],  # denseDir, atlasDir
                        self.spacing.value)

    slicer.mrmlScene.RemoveNode(atlasModel); slicer.mrmlScene.RemoveNode(atlasLMs)
    self.log.appendPlainText(f"Done. Output: {F['output']} (atlas/, dense/, alignedModels/, alignedLMs/)")

  def _generateAtlas(self, F, useSimilarity):
      logic=BUILDERLogic()
      closest_key = logic.getClosestToMeanPath(F['originalLMs'])
      lm_path = logic._resolve_by_key(F['originalLMs'], closest_key, ('.mrk.json','.fcsv'))
      if not lm_path: raise ValueError(f"Could not resolve LM file for key '{closest_key}'")
      baseLM = slicer.util.loadMarkups(lm_path)
      baseModel = logic.getModelFileByID(F['originalModels'], closest_key)
      if not baseModel: raise ValueError(f"Could not resolve model for key '{closest_key}'")
      self.log.appendPlainText(f"Closest sample to mean: {closest_key}")
      logic.runAlign(baseModel, baseLM, F['originalModels'], F['originalLMs'], F['alignedModels'], F['alignedLMs'], useSimilarity)
      self.log.appendPlainText("Building mean atlas")
      atlasModel, atlasLMs = logic.runMean(F['alignedLMs'], F['alignedModels'])
      slicer.mrmlScene.RemoveNode(baseModel); slicer.mrmlScene.RemoveNode(baseLM)
      return atlasModel, atlasLMs


class BUILDERLogic(ScriptedLoadableModuleLogic):

  def numpyToFiducialNode(self, arr, name):
    n=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',name)
    slicer.util.updateMarkupsControlPointsFromArray(n, arr)
    return n

  def fiducialNodeToPolyData(self, nodeOrPath, load=True):
      n = slicer.util.loadMarkups(nodeOrPath) if isinstance(nodeOrPath,str) else nodeOrPath
      arr = slicer.util.arrayFromMarkupsControlPoints(n).astype(np.float32, copy=False)
      pts = vtk.vtkPoints(); pts.SetData(vtk_np.numpy_to_vtk(arr, deep=True))
      pd = vtk.vtkPolyData(); pd.SetPoints(pts)
      if load and isinstance(nodeOrPath,str): slicer.mrmlScene.RemoveNode(n)
      return pd

  def importLandmarks(self, topDir):
      g = vtk.vtkMultiBlockDataGroupFilter(); names=[]
      for f in sorted(os.listdir(topDir)):
          fl = f.lower()
          if fl.endswith((".fcsv",".mrk.json")):
              stem = Path(f)
              while stem.suffix in {'.fcsv','.mrk','.json'}: stem = stem.with_suffix('')
              names.append(stem.name)
              poly = self.fiducialNodeToPolyData(os.path.join(topDir,f), load=True)
              g.AddInputData(poly)
      g.Update()
      return names, g.GetOutput()


  def importMeshes(self, topDir, exts):
      g = vtk.vtkMultiBlockDataGroupFilter(); names = []
      try:
          slicer.app.setRenderPaused(True)
      except: pass
      for f in sorted(os.listdir(topDir)):
          fl = f.lower()
          if not fl.endswith(tuple(exts)): continue
          path = os.path.join(topDir, f)
          base, _ = os.path.splitext(f)
          node = slicer.util.loadModel(path)       # loads into MRML
          if not node: continue
          pd = vtk.vtkPolyData(); pd.DeepCopy(node.GetPolyData())  # detach from MRML/display state
          g.AddInputData(pd); names.append(base)
          slicer.mrmlScene.RemoveNode(node)        # keep the scene clean
      g.Update()
      try:
          slicer.app.setRenderPaused(False)
      except: pass
      return names, g.GetOutput()

  def procrustesImposition(self, originalLandmarks, rigidOnly):
    flt=vtk.vtkProcrustesAlignmentFilter()
    if rigidOnly: flt.GetLandmarkTransform().SetModeToRigidBody()
    flt.SetInputData(originalLandmarks); flt.Update()
    return flt.GetMeanPoints(), flt.GetOutput()
  
  def getClosestToMeanIndex(self, meanShape, alignedPoints):
      N = alignedPoints.GetNumberOfBlocks()
      if N == 0: return 0
      mean_np = vtk_np.vtk_to_numpy(meanShape.GetData()).astype(np.float32, copy=False)

      best = 0; best_val = float('inf')
      for i in range(N):
          blk = alignedPoints.GetBlock(i)
          if not blk or blk.GetNumberOfPoints()==0: continue
          a = vtk_np.vtk_to_numpy(blk.GetPoints().GetData()).astype(np.float32, copy=False)
          # sum of squared distances (no sqrt)
          v = np.sum((mean_np - a)**2)
          if v < best_val: best_val, best = v, i
      return best

  def getClosestToMeanPath(self, landmarkDirectory):
    names,L=self.importLandmarks(landmarkDirectory); mean,AL=self.procrustesImposition(L, False)
    idx=self.getClosestToMeanIndex(mean, AL)
    return names[idx]

  def runMean(self, lmDir, meshDir):
      names, Ms, _, LMs = self.importPaired(meshDir, lmDir)
      G, baseIdx = self.denseCorrespondence(LMs, Ms)
      avg = self.computeAverageModelFromGroup(G, baseIdx)
      M = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode','Atlas Model'); M.CreateDefaultDisplayNodes(); M.SetAndObservePolyData(avg)
      L = self.computeAverageLM(LMs); L.GetDisplayNode().SetPointLabelsVisibility(False)
      return M, L

  def computeAverageLM(self, fidGroup):
      n=fidGroup.GetNumberOfBlocks(); m=fidGroup.GetBlock(0).GetNumberOfPoints()
      arr=np.empty((m,3,n), dtype=np.float32)
      for i in range(n):
          pts=fidGroup.GetBlock(i).GetPoints().GetData()
          arr[:,:,i]=vtk_np.vtk_to_numpy(pts)
      mean=np.mean(arr,axis=2)
      return self.numpyToFiducialNode(mean,"Atlas Landmarks")
  
  def _points_from_markups(self, node):
      arr = slicer.util.arrayFromMarkupsControlPoints(node).astype(np.float32, copy=False)
      pts = vtk.vtkPoints(); pts.SetData(vtk_np.numpy_to_vtk(arr, deep=True))
      return pts
  

  def runAlign(self, baseMeshNode, baseLMNode, meshDir, lmDir, outMeshDir, outLMDir,
              useSimilarity, slmDir=False, outSLMDir=False, progress=None):
      semis = bool(slmDir and outSLMDir)
      scratch = None; model_scratch = None

      # Build paired key set up front
      md = self._list_keys(meshDir, ('.ply', '.stl', '.vtp', '.vtk', '.obj'))
      ld = self._list_keys(lmDir,  ('.mrk.json', '.fcsv'))
      keys = sorted(set(md) & set(ld))
      total = len(keys); done = 0

      try:
          try: slicer.app.setRenderPaused(True)
          except Exception: pass

          scratch = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'scratch_lm')
          model_scratch = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'scratch_model')
          scratch.SetHideFromEditors(True); model_scratch.SetHideFromEditors(True)
          if scratch.GetDisplayNode(): 
              scratch.GetDisplayNode().SetVisibility(False); scratch.GetDisplayNode().SetPointLabelsVisibility(False)
          if model_scratch.GetDisplayNode(): 
              model_scratch.GetDisplayNode().SetVisibility(False)

          tgt_pts = self._points_from_markups(baseLMNode)
          tf = vtk.vtkLandmarkTransform(); tpf = vtk.vtkTransformPolyDataFilter()

          for sid in keys:
              # Load paired inputs
              lmNode = self.getLandmarkFileByID(lmDir, sid)
              modelNode = slicer.util.loadModel(md[sid])
              mesh_pd = vtk.vtkPolyData(); mesh_pd.DeepCopy(modelNode.GetPolyData())
              slicer.mrmlScene.RemoveNode(modelNode)

              if lmNode.GetNumberOfControlPoints() != baseLMNode.GetNumberOfControlPoints():
                  slicer.mrmlScene.RemoveNode(lmNode)
                  raise ValueError(f"Landmark mismatch for {sid}")

              src_pts = self._points_from_markups(lmNode)
              tf.SetModeToSimilarity() if useSimilarity else tf.SetModeToRigidBody()
              tf.SetSourceLandmarks(src_pts); tf.SetTargetLandmarks(tgt_pts); tf.Update()

              tpf.SetInputData(mesh_pd); tpf.SetTransform(tf); tpf.Update()
              aligned_mesh_pd = tpf.GetOutput()

              with slicer.util.NodeModify(model_scratch):
                  model_scratch.SetAndObservePolyData(aligned_mesh_pd)
              slicer.util.saveNode(model_scratch, os.path.join(outMeshDir, f"{sid}_align.ply"))

              lm = slicer.util.arrayFromMarkupsControlPoints(lmNode).astype(np.float32, copy=False)
              lm_h = np.hstack((lm, np.ones((lm.shape[0], 1), dtype=np.float32)))
              M = vtk.vtkMatrix4x4(); tf.GetMatrix(M)
              Mn = np.array([[M.GetElement(r,c) for c in range(4)] for r in range(4)], dtype=np.float32)
              aligned = (Mn @ lm_h.T).T[:, :3]

              with slicer.util.NodeModify(scratch):
                  slicer.util.updateMarkupsControlPointsFromArray(scratch, aligned)
              slicer.util.saveNode(scratch, os.path.join(outLMDir, f"{sid}_align.mrk.json"))
              slicer.mrmlScene.RemoveNode(lmNode)

              if semis:
                  slm = self.getLandmarkFileByID(slmDir, sid)
                  if slm:
                      slm_arr = slicer.util.arrayFromMarkupsControlPoints(slm).astype(np.float32, copy=False)
                      slm_h = np.hstack((slm_arr, np.ones((slm_arr.shape[0], 1), dtype=np.float32)))
                      slm_al = (Mn @ slm_h.T).T[:, :3]
                      with slicer.util.NodeModify(scratch):
                          slicer.util.updateMarkupsControlPointsFromArray(scratch, slm_al)
                      slicer.util.saveNode(scratch, os.path.join(outSLMDir, f"{sid}_align.mrk.json"))
                      slicer.mrmlScene.RemoveNode(slm)

              done += 1
              if progress:
                  progress(done, total, sid)
                  qt.QApplication.processEvents()

      finally:
          if scratch: slicer.mrmlScene.RemoveNode(scratch)
          if model_scratch: slicer.mrmlScene.RemoveNode(model_scratch)
          try: slicer.app.setRenderPaused(False)
          except Exception: pass


  def denseCorrespondence(self, LMs, meshes):
      mean, AL = self.procrustesImposition(LMs, False)
      N = AL.GetNumberOfBlocks()
      if N == 0:
          out = vtk.vtkMultiBlockDataSet()
          return out, 0

      baseIdx = self.getClosestToMeanIndex(mean, AL)
      baseMesh = meshes.GetBlock(baseIdx)
      baseLM   = LMs.GetBlock(baseIdx).GetPoints()
      if baseMesh is None or baseLM is None:
        raise ValueError("denseCorrespondence: invalid base mesh/landmarks.")

      tri = vtk.vtkTriangleFilter()
      tri.SetInputData(baseMesh); tri.PassLinesOff(); tri.PassVertsOff(); tri.Update()
      baseTri   = tri.GetOutput()
      basePolys = baseTri.GetPolys()

      # Warp the base to mean to get the template points used for NN search
      t2 = vtk.vtkThinPlateSplineTransform()
      t2.SetSourceLandmarks(baseLM); t2.SetTargetLandmarks(mean); t2.SetBasisToR()
      f2 = vtk.vtkTransformPolyDataFilter()
      f2.SetInputData(baseTri); f2.SetTransform(t2); f2.Update()
      bWarpPts = f2.GetOutput().GetPoints()

      grp = vtk.vtkMultiBlockDataGroupFilter()
      for i in range(N):
          lm_i = LMs.GetBlock(i).GetPoints()
          corr = self._corr_pair_kdtree(meshes.GetBlock(i), lm_i, bWarpPts, basePolys, mean)
          grp.AddInputData(corr)
      grp.Update()
      return grp.GetOutput(), baseIdx

  
  def _corr_pair_kdtree(self, mesh, lm, bWarpPts, basePolys, mean, specimen_name="Unknown"):
      """
      Finds dense correspondence using a fast k-d tree search.
      This version contains the corrected filtering logic.
      """
      t1 = vtk.vtkThinPlateSplineTransform(); t1.SetSourceLandmarks(lm); t1.SetTargetLandmarks(mean); t1.SetBasisToR()
      f1 = vtk.vtkTransformPolyDataFilter(); f1.SetInputData(mesh); f1.SetTransform(t1); f1.Update()
      mWarp = f1.GetOutput()

      warped_specimen_pts = vtk_np.vtk_to_numpy(mWarp.GetPoints().GetData())
      warped_template_pts = vtk_np.vtk_to_numpy(bWarpPts.GetData())

      finite_mask = np.all(np.isfinite(warped_specimen_pts), axis=1)
      if not np.all(finite_mask):
          print(f"WARNING: Specimen '{specimen_name}' has non-finite points after TPS warp. Filtering them out.")
          clean_warped_specimen_pts = warped_specimen_pts[finite_mask]
      else:
          clean_warped_specimen_pts = warped_specimen_pts

      tree = cKDTree(clean_warped_specimen_pts) 

      try:
        distances, indices_in_clean_set = tree.query(warped_template_pts, k=1, workers=-1)
      except TypeError:
        distances, indices_in_clean_set = tree.query(warped_template_pts, k=1)

      corresponding_pts_np = clean_warped_specimen_pts[indices_in_clean_set] 

      pts_vtk = vtk.vtkPoints()
      pts_vtk.SetData(vtk_np.numpy_to_vtk(corresponding_pts_np, deep=True))
      corr = vtk.vtkPolyData(); corr.SetPoints(pts_vtk); corr.SetPolys(basePolys)

      inv = vtk.vtkThinPlateSplineTransform(); inv.SetSourceLandmarks(mean); inv.SetTargetLandmarks(lm); inv.SetBasisToR()
      f3 = vtk.vtkTransformPolyDataFilter(); f3.SetInputData(corr); f3.SetTransform(inv); f3.Update()
      return f3.GetOutput()
    
  def sparseCorrespondenceBaseMesh(self, LMs, meshes, atlasMesh, atlasLM, keep_idx):
    bPts=atlasMesh.GetPoints(); qpts=vtk.vtkPoints(); qpts.SetNumberOfPoints(len(keep_idx))
    p=[0,0,0]
    for i,k in enumerate(keep_idx): bPts.GetPoint(k,p); qpts.SetPoint(i,p)
    out=[]
    N=LMs.GetNumberOfBlocks()
    for i in range(N):
        t=vtk.vtkThinPlateSplineTransform(); t.SetSourceLandmarks(LMs.GetBlock(i).GetPoints()); t.SetTargetLandmarks(atlasLM); t.SetBasisToR()
        f=vtk.vtkTransformPolyDataFilter(); f.SetInputData(meshes.GetBlock(i)); f.SetTransform(t); f.Update(); mWarp=f.GetOutput()
        tri=vtk.vtkTriangleFilter(); tri.SetInputData(mWarp); tri.PassLinesOff(); tri.PassVertsOff(); tri.Update(); mWarp=tri.GetOutput()
        loc=vtk.vtkStaticCellLocator(); loc.SetDataSet(mWarp); loc.BuildLocator()
        outPts=vtk.vtkPoints(); outPts.SetNumberOfPoints(len(keep_idx))
        q=[0,0,0]; cid=vtk.reference(0); sid=vtk.reference(0); d2=vtk.reference(0.0)
        for j in range(qpts.GetNumberOfPoints()):
            qpts.GetPoint(j,p); loc.FindClosestPoint(p,q,cid,sid,d2); outPts.SetPoint(j,q)
        corr=vtk.vtkPolyData(); corr.SetPoints(outPts)
        inv=vtk.vtkThinPlateSplineTransform(); inv.SetSourceLandmarks(atlasLM); inv.SetTargetLandmarks(LMs.GetBlock(i).GetPoints()); inv.SetBasisToR()
        f2=vtk.vtkTransformPolyDataFilter(); f2.SetInputData(corr); f2.SetTransform(inv); f2.Update()
        arr=vtk_np.vtk_to_numpy(f2.GetOutput().GetPoints().GetData()).astype(np.float32,copy=False)
        out.append(arr)
    return out


  def computeAverageModelFromGroup(self, grp, baseIdx):
      n=grp.GetNumberOfBlocks(); m=grp.GetBlock(0).GetNumberOfPoints()
      arr=np.empty((m,3,n), dtype=np.float32)
      base=grp.GetBlock(baseIdx)
      for i in range(n):
          arr[:,:,i]=vtk_np.vtk_to_numpy(grp.GetBlock(i).GetPoints().GetData())
      mean=np.mean(arr,axis=2)
      pts=self.convertPointsToVTK(mean)
      out=vtk.vtkPolyData(); out.SetPoints(pts.GetPoints()); out.SetPolys(base.GetPolys())
      return out

  def convertPointsToVTK(self, pts):
    a=vtk_np.numpy_to_vtk(pts, deep=True, array_type=vtk.VTK_FLOAT); p=vtk.vtkPoints(); p.SetData(a); poly=vtk.vtkPolyData(); poly.SetPoints(p); return poly
  

  def exportDenseLMs(self, atlasModelNode, atlasLMNode,
                    alignedModelsDir, alignedLMsDir,
                    denseDir, atlasDir,
                    spacingTolerance):

      def _has_files(d, exts):
          try:
              return any(fn.lower().endswith(exts) for fn in os.listdir(d))
          except Exception:
              return False

      if not _has_files(alignedModelsDir, ('.ply', '.stl', '.vtp', '.vtk', '.obj')) \
        or not _has_files(alignedLMsDir, ('.mrk.json', '.fcsv', '.json')):
          raise ValueError("No aligned models/landmarks found to export.")

      try:
          slicer.app.setRenderPaused(True)
      except Exception:
          pass

      spacingFrac = float(spacingTolerance) / 100.0
      keep_idx = self.sample_indices_poisson(atlasModelNode.GetPolyData(), spacingFrac)
      if keep_idx.size == 0:
        raise ValueError("Sampling radius produced 0 template points. Reduce the radius and try again.")
      baseLM_pts = self.fiducialNodeToPolyData(atlasLMNode, load=False).GetPoints()

      names, models, _, lms = self.importPaired(alignedModelsDir, alignedLMsDir)
      if len(names) == 0:
          raise ValueError("No paired aligned models/landmarks found to export.")

      pts_list = self.sparseCorrespondenceBaseMesh(
          lms, models, atlasModelNode.GetPolyData(), baseLM_pts, keep_idx
      )

      scratch = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'scratch_export')
      scratch.SetHideFromEditors(True)
      sd = scratch.GetDisplayNode()
      if sd:
          sd.SetVisibility(False); sd.SetPointLabelsVisibility(False)

      # Specimen dense correspondences -> dense/
      for i, arr in enumerate(pts_list):
          with slicer.util.NodeModify(scratch):
              slicer.util.updateMarkupsControlPointsFromArray(scratch, arr)
          slicer.util.saveNode(scratch, os.path.join(denseDir, f"{names[i]}.mrk.json"))

      # Sampled atlas template points -> atlas/ (atlas.mrk.json)
      atlas_all = vtk_np.vtk_to_numpy(atlasModelNode.GetPolyData().GetPoints().GetData())
      atlas_tpl = atlas_all[keep_idx].astype(np.float32, copy=False)
      with slicer.util.NodeModify(scratch):
          slicer.util.updateMarkupsControlPointsFromArray(scratch, atlas_tpl)
      slicer.util.saveNode(scratch, os.path.join(atlasDir, "atlas.mrk.json"))

      slicer.mrmlScene.RemoveNode(scratch)

      try:
          slicer.app.setRenderPaused(False)
      except Exception:
          pass


  def saveAtlasOnly(self, atlasModelNode, atlasLMNode, atlasDir):
    # atlas mesh and sparse (hand) landmarks
    slicer.util.saveNode(atlasModelNode, os.path.join(atlasDir, "atlas.ply"))
    slicer.util.saveNode(atlasLMNode,   os.path.join(atlasDir, "atlasLM.mrk.json"))

  # Add this helper in BUILDERLogic
  def _key_no_ext(self, f):
      s = Path(f)
      while s.suffix.lower() in ('.json','.mrk','.fcsv'): s = s.with_suffix('')
      if s.suffix.lower() in ('.ply','.stl','.vtp','.vtk','.obj'): s = s.with_suffix('')
      return s.name

  def _resolve_by_key(self, directory, key, exts):
      kl = str(key).lower()
      for f in os.listdir(directory):
          if self._key_no_ext(f).lower() == kl:
              fl = f.lower()
              for ext in exts:
                  if fl.endswith(ext): return os.path.join(directory,f)
              return os.path.join(directory,f)
      return None

  def getModelFileByID(self, directory, subjectID):
      p = self._resolve_by_key(directory, subjectID, ('.ply','.stl','.vtp','.vtk','.obj'))
      return slicer.util.loadModel(p) if p else None

  def getLandmarkFileByID(self, directory, subjectID):
      p = self._resolve_by_key(directory, subjectID, ('.mrk.json','.fcsv'))
      return slicer.util.loadMarkups(p) if p else None
  
  def _list_keys(self, d, ok_exts):
    out={}
    for f in os.listdir(d):
        fl=f.lower()
        if fl.endswith(tuple(ok_exts)):
            out[self._key_no_ext(f).lower()] = os.path.join(d,f)
    return out

  def importPaired(self, modelsDir, lmsDir):
      md = self._list_keys(modelsDir, ('.ply','.stl','.vtp','.vtk','.obj'))
      ld = self._list_keys(lmsDir,    ('.mrk.json','.fcsv'))
      keys = sorted(set(md) & set(ld))
      gm = vtk.vtkMultiBlockDataGroupFilter()
      gl = vtk.vtkMultiBlockDataGroupFilter()
      names=[]
      for k in keys:
          mnode = slicer.util.loadModel(md[k]); mpd = vtk.vtkPolyData(); mpd.DeepCopy(mnode.GetPolyData()); slicer.mrmlScene.RemoveNode(mnode)
          gm.AddInputData(mpd)
          lnode = slicer.util.loadMarkups(ld[k]); gl.AddInputData(self.fiducialNodeToPolyData(lnode, load=False)); slicer.mrmlScene.RemoveNode(lnode)
          names.append(k)
      gm.Update(); gl.Update()
      return names, gm.GetOutput(), names, gl.GetOutput()
  def _bbox_diag(self, pd):
    b = pd.GetBounds()
    return float(np.linalg.norm([b[1]-b[0], b[3]-b[2], b[5]-b[4]]))

  def sample_indices_poisson(self, pd, frac):
      # Try VTK Poisson disk with ID preservation
      try:
          idf = vtk.vtkIdFilter(); idf.PointIdsOn(); idf.SetPointIdsArrayName("origIds")
          idf.SetInputData(pd); idf.Update()
          rad = max(1e-9, float(frac)) * self._bbox_diag(pd)
          ps = vtk.vtkPoissonDiskSampler(); ps.SetRadius(rad)
          ps.SetInputConnection(idf.GetOutputPort()); ps.Update()
          arr = ps.GetOutput().GetPointData().GetArray("origIds")
          if arr: return vtk_np.vtk_to_numpy(arr).astype(np.int64, copy=False)
      except Exception:
          pass
      # Fallback: greedy radius-based suppression using cKDTree (preserves original indices)
      pts = vtk_np.vtk_to_numpy(pd.GetPoints().GetData())
      n = pts.shape[0]
      if n == 0: return np.empty((0,), np.int64)
      r = max(1e-9, float(frac)) * self._bbox_diag(pd)
      kdt = cKDTree(pts)
      remaining = np.ones(n, dtype=bool)
      out = []
      # seed far from centroid to reduce bias
      c = pts.mean(axis=0)
      i0 = int(np.argmax(np.sum((pts - c)**2, axis=1)))
      stack = [i0]
      while stack:
          i = stack.pop()
          if not remaining[i]: continue
          out.append(i)
          nbrs = kdt.query_ball_point(pts[i], r)
          remaining[nbrs] = False
          # pick next uncovered point if any
          if remaining.any():
              nxt = int(np.flatnonzero(remaining)[0]) # cheap; order doesn’t matter much here
              stack.append(nxt)
      out.sort()
      return np.array(out, dtype=np.int64)
  
  def previewCountForRadius(self, polyDataOrNode, spacingPct):
    pd = polyDataOrNode if isinstance(polyDataOrNode, vtk.vtkPolyData) else polyDataOrNode.GetPolyData()
    keep_idx = self.sample_indices_poisson(pd, float(spacingPct)/100.0)
    return int(keep_idx.size), int(pd.GetNumberOfPoints())


    
