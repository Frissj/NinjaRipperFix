bl_info = {
    "name": "NinjaFix Aligner (multi-capture)",
    "author": "Gemini/Google & User",
    "version": (5, 0, 0),
    "blender": (3, 0, 0),
    "category": "Object",
    "description": "Align any number of Ninja Ripper captures with a Remix reference.",
    "warning": "Requires NumPy. First run auto-installer in the panel if necessary.",
}

import bpy
import bmesh
import sys, subprocess
from bpy.props import PointerProperty, StringProperty, BoolProperty, IntProperty
from bpy.types import PropertyGroup, Operator, Panel, UIList
import json
from mathutils import Vector, Matrix, kdtree
import hashlib
import os
import numpy as np
from bpy.props import CollectionProperty
from bpy.types import OperatorFileListElement

# --- Globals ---
CAPTURE_SETS = []
INITIAL_ALIGN_MATRIX = None
GOLDEN_TARGETS = {} # NEW: To store the 'snapshot' of target points
PREFERRED_ANCHOR_HASHES = set()

# --- Dependency Check ---
try:
    import numpy as np
    NUMPY_INSTALLED = True
except ImportError:
    np = None
    NUMPY_INSTALLED = False

# --- Globals ---
CAPTURE_SETS = []
INITIAL_ALIGN_MATRIX = None

# =================================================================================================
# Utility Helpers
# =================================================================================================
def is_ninja(ob):
    """
    True for every mesh imported by Ninja Ripper. Handles ".001" suffixes.
    """
    return ob and ob.type == 'MESH' and ob.name.startswith("mesh_")

def is_remix(ob):
    """
    True for Remix reference parts (meshes not from Ninja Ripper with a dot in the name).
    """
    return ob and ob.type == 'MESH' and ('.' in ob.name) and not ob.name.startswith("mesh_")

def current_ninja_set(scene):
    return {ob.name for ob in scene.objects if is_ninja(ob)}

def mat_to_list(M):
    return [c for r in M for c in r]

def get_db_path():
    """Return path to the per-.blend NinjaFix JSON DB (alongside the .blend)."""
    blend_path = bpy.data.filepath
    if not blend_path:
        return None
    return os.path.join(os.path.dirname(blend_path), ".ninjafix_db.json")

def load_db():
    path = get_db_path()
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_db(db):
    path = get_db_path()
    if not path:
        return
    try:
        with open(path, 'w') as f:
            json.dump(db, f, indent=4)
    except Exception as e:
        print(f"Failed to save NinjaFix DB: {e}")

# --- Forward declaration for PropertyGroup update function ---
def toggle_auto_capture_timer(self, context):
    """Starts or stops the auto-capture modal timer based on the checkbox state."""
    if self.auto_capture:
        # Call the operator to start the timer.
        # It's safe to call this even if it's already running, though it's best if the user doesn't thrash the checkbox.
        bpy.ops.nfix.auto_capture_timer('INVOKE_DEFAULT')
    # If self.auto_capture is False, the running modal operator will detect this on its next tick and terminate itself.

# =================================================================================================
# Core Alignment and Data Logic (NEW AND UNIFIED)
# =================================================================================================
class NFIX_AlignmentLogic:
    """A collection of static methods for consistent alignment logic."""

    @staticmethod
    def get_evaluated_world_points(obj, depsgraph):
        """
        Safely gets the world-space vertex coordinates of an object's final,
        evaluated geometry. Returns an empty array if anything fails.
        """
        if obj is None or obj.type != 'MESH':
            return np.array([])
        
        try:
            eval_obj = obj.evaluated_get(depsgraph)
            # Using .to_mesh() is sufficient and safer inside a try-finally block.
            temp_mesh = eval_obj.to_mesh()
            
            if not temp_mesh.vertices:
                return np.array([])

            # Get points and transform to world space
            size = len(temp_mesh.vertices)
            points = np.empty(size * 3, dtype=np.float32)
            temp_mesh.vertices.foreach_get("co", points)
            points.shape = (size, 3)
            
            # Apply world matrix transformation
            matrix = np.array(eval_obj.matrix_world)
            points_h = np.hstack([points, np.ones((size, 1))])
            world_points = (points_h @ matrix.T)[:, :3]

            return world_points

        finally:
            if 'temp_mesh' in locals() and temp_mesh:
                # This ensures the temp mesh is always freed.
                eval_obj.to_mesh_clear()

    @staticmethod
    def solve_alignment(source_points, target_points):
        """
        Calculates the best-fit AFFINE transform (including non-uniform scale)
        to align source_points to target_points using the least-squares method.
        Returns a 4x4 Blender Matrix.
        """
        if source_points.shape[0] == 0 or source_points.shape[0] != target_points.shape[0]:
            return Matrix.Identity(4)

        n = source_points.shape[0]

        # 1. Pad the source points to homogeneous coordinates
        S_h = np.hstack([source_points, np.ones((n, 1))])

        # 2. Solve the least squares problem T = S_h @ M_T for the 4x3 transform matrix
        M_T, _, _, _ = np.linalg.lstsq(S_h, target_points, rcond=None)

        # 3. Create the 4x4 Blender matrix
        M = Matrix.Identity(4)
    
        # 4. Transpose and assign the result to the Blender matrix using loops
        M_T_transposed = M_T.T
        for r in range(3):
            for c in range(4):
                M[r][c] = M_T_transposed[r, c]

        return M

    @staticmethod
    def get_texture_hash(obj):
        if not obj: return ""
        paths = set()
        for slot in obj.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes: continue
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and getattr(node, 'image', None):
                    paths.add(bpy.path.abspath(node.image.filepath))
        m = hashlib.md5()
        for p in sorted(list(paths)):
            try:
                with open(p, 'rb') as f: m.update(f.read())
            except Exception: m.update(p.encode('utf-8'))
        return m.hexdigest()

class NFIX_OT_RemoveMatDuplicates(Operator):
    bl_idname = "nfix.remove_mat_duplicates"
    bl_label  = "Remove mat_ Duplicates"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return NUMPY_INSTALLED

    def execute(self, context):
        """
        Deletes meshes that are duplicates in GEOMETRY AND LOCATION (World Space),
        with caching of evaluated points and KD-trees, and never deleting any mesh
        that has a “normal” material (i.e., any material whose name does NOT start with 'mat_').
        If a cluster has both normal-material and mat_-only meshes, delete all mat_-only.
        If cluster all mat_-only: pick one keeper (pure Ninja first, else first) and delete others.
        If cluster all normal-material: delete none.
        """
        import numpy as np
        from mathutils import Matrix, kdtree

        depsgraph = context.evaluated_depsgraph_get()
        scene = context.scene
        tol = 0.005

        print("\n================ NinjaFix DEBUG START ================")
        print(f"[DEBUG] Tolerance for duplicate detection: {tol}")

        # ---------------------------------------------------------------------
        # 1) PRECOMPUTE & CACHE: world-space points, KD-trees, has_normal flag
        # ---------------------------------------------------------------------
        world_pts_cache = {}
        kd_cache        = {}
        entries         = []
        matflag_cache   = {}

        def has_normal_mat(obj):
            """
            Returns True if object has any material slot whose material name
            does NOT start with 'mat_', or if it has no material slots or slot.material is None.
            Returns False only if there is at least one material slot and every slot.material.name starts with 'mat_'.
            """
            name = obj.name
            if name in matflag_cache:
                return matflag_cache[name]
            # If no material slots, treat as normal
            if not obj.material_slots:
                matflag_cache[name] = True
                return True
            # Check each slot
            for slot in obj.material_slots:
                mat = slot.material
                if mat is None:
                    matflag_cache[name] = True
                    return True
                mname = mat.name
                if not mname.lower().startswith("mat_"):
                    matflag_cache[name] = True
                    return True
            # All slots have material names starting with 'mat_'
            matflag_cache[name] = False
            return False

        # collect meshes and build caches
        for ob in list(scene.objects):
            if ob.type != 'MESH':
                continue

            # cache world points
            pts = world_pts_cache.get(ob.name)
            if pts is None:
                pts = NFIX_AlignmentLogic.get_evaluated_world_points(ob, depsgraph)
                world_pts_cache[ob.name] = pts

            vcount = pts.shape[0]
            print(f"[DEBUG] Object '{ob.name}': vertex count = {vcount}")
            if vcount == 0:
                continue

            # build KD-tree once and cache it
            kd = kd_cache.get(ob.name)
            if kd is None:
                kd = kdtree.KDTree(vcount)
                for idx, co in enumerate(pts):
                    kd.insert(co, idx)
                kd.balance()
                kd_cache[ob.name] = kd

            # determine has_normal and optionally print material info
            has_norm = has_normal_mat(ob)
            mat_names = [slot.material.name if slot.material else "None" for slot in ob.material_slots]
            print(f"[DEBUG]   Materials on '{ob.name}': {mat_names}")
            print(f"[DEBUG]   has_normal flag = {has_norm}")

            entries.append({
                'obj': ob,
                'pts': pts,
                'vcount': vcount,
                'kd': kd,
                'has_normal': has_norm,
            })

        n = len(entries)
        print(f"[DEBUG] Total meshes considered: {n}")
        if n < 2:
            self.report({'WARNING'}, "Not enough meshes to compare.")
            print("================ NinjaFix DEBUG END ================\n")
            return {'CANCELLED'}

        # ---------------------------------------------------------------------
        # 2) GROUP BY vertex count to skip mismatches
        # ---------------------------------------------------------------------
        groups = {}
        for idx, ent in enumerate(entries):
            groups.setdefault(ent['vcount'], []).append(idx)

        # ---------------------------------------------------------------------
        # 3) PAIRWISE KD-CHECK WITH CACHED TREES
        # ---------------------------------------------------------------------
        adj = {i: set() for i in range(n)}
        for vcount, idxs in groups.items():
            if len(idxs) < 2:
                continue
            print(f"[DEBUG] Comparing {len(idxs)} meshes with {vcount} verts each")
            for ii in range(len(idxs)):
                i = idxs[ii]
                pts_i = entries[i]['pts']
                name_i = entries[i]['obj'].name
                for jj in range(ii + 1, len(idxs)):
                    j = idxs[jj]
                    name_j = entries[j]['obj'].name
                    print(f"\n[DEBUG] Comparing '{name_i}' vs '{name_j}'")
                    kd_j = entries[j]['kd']
                    max_err = 0.0
                    for co in pts_i:
                        _, _, d = kd_j.find(co)
                        if d > max_err:
                            max_err = d
                            if max_err >= tol:
                                print(f"[DEBUG] Early exit: d={d:.6f} >= tol")
                                break
                    print(f"[DEBUG] Max NN distance = {max_err:.6f}")
                    if max_err < tol:
                        adj[i].add(j)
                        adj[j].add(i)
                        print(f"[DEBUG] Marking duplicates (within tol)")

        # ---------------------------------------------------------------------
        # 4) BUILD CLUSTERS
        # ---------------------------------------------------------------------
        visited = set()
        clusters = []
        for i in range(n):
            if i in visited:
                continue
            stack = [i]
            comp = set()
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                visited.add(u)
                for v in adj[u]:
                    if v not in comp:
                        stack.append(v)
            clusters.append(comp)
        print(f"[DEBUG] Found {len(clusters)} clusters:")
        for ci, comp in enumerate(clusters):
            names = [entries[k]['obj'].name for k in comp]
            print(f"  Cluster {ci}: {names}")

        # ---------------------------------------------------------------------
        # 5) SELECT KEEPERS & COLLECT FOR DELETION
        #    - Never delete any mesh with has_normal=True.
        #    - If cluster has both normal-material and mat_-only meshes: delete all mat_-only.
        #    - If cluster all mat_-only: pick one keeper (pure Ninja first, else first) and delete others.
        #    - If cluster all normal-material: delete none.
        # ---------------------------------------------------------------------
        to_delete = []
        for ci, comp in enumerate(clusters):
            if len(comp) < 2:
                continue
            ents = [entries[k] for k in comp]
            names = [e['obj'].name for e in ents]
            print(f"\n[DEBUG] Processing Cluster {ci}: {names}")

            normal_objs = [e for e in ents if e['has_normal']]
            matonly_objs = [e for e in ents if not e['has_normal']]

            if normal_objs:
                # Keep all normal-material meshes; delete all mat_-only meshes
                print(f"[DEBUG] Cluster {ci} has {len(normal_objs)} normal-material meshes; preserving them.")
                if matonly_objs:
                    for e in matonly_objs:
                        ob_e = e['obj']
                        print(f"[DEBUG] Marking mat_-only '{ob_e.name}' for deletion")
                        to_delete.append(ob_e)
                else:
                    print(f"[DEBUG] No mat_-only meshes in Cluster {ci}; nothing to delete.")
            else:
                # All are mat_-only: pick keeper among them
                keeper = None
                reason = ""
                # 1) pure Ninja
                for e in ents:
                    nm = e['obj'].name
                    if nm.startswith("mesh_") and "." not in nm:
                        keeper = e['obj']
                        reason = "pure Ninja"
                        break
                if keeper is None:
                    keeper = ents[0]['obj']
                    reason = "first fallback"
                print(f"[DEBUG] Keeper among mat_-only Cluster {ci}: '{keeper.name}' ({reason})")
                for e in ents:
                    ob_e = e['obj']
                    if ob_e is not keeper:
                        print(f"[DEBUG] Marking mat_-only '{ob_e.name}' for deletion")
                        to_delete.append(ob_e)

        # ---------------------------------------------------------------------
        # 6) BATCH DELETE VIA OPERATOR
        # ---------------------------------------------------------------------
        # Deselect everything first
        bpy.ops.object.select_all(action='DESELECT')

        # Select all duplicates
        for ob in to_delete:
            ob.select_set(True)

        # Set one as the active object (required by the operator)
        if to_delete:
            context.view_layer.objects.active = to_delete[0]

        # Delete all selected at once
        bpy.ops.object.delete()

        # Report how many were removed
        self.report({'INFO'}, f"Removed {len(to_delete)} duplicates.")
        print("================ NinjaFix DEBUG END ================\n")
        return {'FINISHED'}

# =================================================================================================
# Property Groups & UI Setup
# =================================================================================================
class NinjaFixSettings(PropertyGroup):
    source_obj: PointerProperty(type=bpy.types.Object)
    target_obj: PointerProperty(type=bpy.types.Object)
    auto_capture: BoolProperty(
        name="Auto Capture",
        description="Automatically add a new capture when the number of mesh objects changes",
        default=False,
        update=toggle_auto_capture_timer
    )
    import_folders: BoolProperty(
        name="Import From Folders",
        description="Show the multi-folder import controls",
        default=False
    )
    flip_geometry: BoolProperty(
        name="Flip Geometry",
        description="Flip geometry on import (use right-handed projection)",
        default=False
    )

class NFIX_OT_InstallNumpy(Operator):
    bl_idname = "nfix.install_numpy"
    bl_label = "Install NumPy"
    def execute(self, context):
        try:
            py = sys.executable
            subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call([py, "-m", "pip", "install", "numpy"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.report({'INFO'}, "NumPy installed! Please restart Blender to enable the addon.")
        except Exception as e:
            self.report({'ERROR'}, f"NumPy installation failed: {e}")
        return {'FINISHED'}

class NFIX_OT_SetObjects(Operator):
    bl_idname = "nfix.set_objects"
    bl_label = "Set from Selection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # This poll correctly checks if two objects are selected
        return len(context.selected_objects) == 2 and context.mode == 'OBJECT'

    def execute(self, context):
        src = tgt = None
        # This logic now correctly reads from the selection
        for ob in context.selected_objects:
            if is_ninja(ob):
                src = ob
            elif is_remix(ob):
                tgt = ob
        
        if not (src and tgt):
            self.report({'ERROR'}, "Selection must include one Ninja mesh and one Remix mesh.")
            return {'CANCELLED'}
            
        st = context.scene.ninjafix_settings
        st.source_obj = src
        st.target_obj = tgt
        self.report({'INFO'}, f"Source: '{src.name}', Target: '{tgt.name}'.")
        return {'FINISHED'}
        
# =================================================================================================
# STAGE 2: CALCULATE AND APPLY INITIAL FIX
# =================================================================================================
class NFIX_OT_AutoCaptureTimer(Operator):
    """Checks for mesh count changes and triggers auto-capture if enabled."""
    bl_idname = "nfix.auto_capture_timer"
    bl_label = "Auto Capture Timer"

    _timer = None
    last_mesh_count: IntProperty(default=-1)

    def modal(self, context, event):
        st = context.scene.ninjafix_settings
        if not st.auto_capture:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            current_mesh_count = len([o for o in context.scene.objects if o.type == 'MESH'])
            if self.last_mesh_count == -1: self.last_mesh_count = current_mesh_count

            if current_mesh_count != self.last_mesh_count:
                ninjas = current_ninja_set(context.scene)
                recorded = set().union(*(c['objects'] for c in CAPTURE_SETS))
                if ninjas - recorded: # Only trigger if there are NEW ninja meshes
                    bpy.ops.nfix.set_current_capture('EXEC_DEFAULT')
                self.last_mesh_count = current_mesh_count
                # Force UI redraw
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for region in area.regions:
                            if region.type == 'UI':
                                region.tag_redraw()
        return {'PASS_THROUGH'}

    def execute(self, context):
        self.last_mesh_count = len([o for o in context.scene.objects if o.type == 'MESH'])
        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
            self._timer = None

class NFIX_OT_CalculateAndBatchFix(Operator):
    bl_idname = "nfix.calculate_and_batch_fix"
    bl_label  = "Generate & Apply Fix"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        st = context.scene.ninjafix_settings
        return NUMPY_INSTALLED and st.source_obj and st.target_obj

    def execute(self, context):
        global INITIAL_ALIGN_MATRIX, GOLDEN_TARGETS, PREFERRED_ANCHOR_HASHES

        st = context.scene.ninjafix_settings
        depsgraph = context.evaluated_depsgraph_get()

        src_pts = NFIX_AlignmentLogic.get_evaluated_world_points(st.source_obj, depsgraph)
        tgt_pts = NFIX_AlignmentLogic.get_evaluated_world_points(st.target_obj, depsgraph)

        if src_pts.shape[0] == 0 or src_pts.shape[0] != tgt_pts.shape[0]:
            self.report({'ERROR'}, "Meshes must be valid and have identical vertex counts.")
            return {'CANCELLED'}

        # --- compute initial alignment matrix ---
        M0 = NFIX_AlignmentLogic.solve_alignment(src_pts, tgt_pts)
        INITIAL_ALIGN_MATRIX = M0.copy()

        # --- extract non-uniform scale and save FULL matrix + scale to DB ---
        loc, rot, scale = M0.decompose()
        db = load_db()
        blend_file = bpy.data.filepath
        db[blend_file] = {
            "scale": [scale.x, scale.y, scale.z],
            "matrix": mat_to_list(M0)
        }
        save_db(db)
        self.report({'INFO'}, f"Saved full alignment matrix and scale to DB for {blend_file}")

        # NEW: Add the initial anchor's hash to the preferred set
        initial_anchor_hash = NFIX_AlignmentLogic.get_texture_hash(st.source_obj)
        if initial_anchor_hash:
            PREFERRED_ANCHOR_HASHES.add(initial_anchor_hash)

        # --- apply M0 to your first capture ---
        first_set = current_ninja_set(context.scene)
        for name in first_set:
            ob = context.scene.objects.get(name)
            if is_ninja(ob):
                ob["ninjafix_prev_matrix"] = mat_to_list(ob.matrix_world)
                ob.matrix_world = M0 @ ob.matrix_world

        # --- cache golden targets (unchanged) ---
        self.report({'INFO'}, "Caching golden target data...")
        GOLDEN_TARGETS.clear()
        depsgraph.update()

        remix_objs = [o for o in context.scene.objects if is_remix(o)]
        cap1_objs = [context.scene.objects.get(n) for n in first_set if context.scene.objects.get(n)]
        for t_obj in remix_objs + cap1_objs:
            if not t_obj or t_obj.type != 'MESH':
                continue
            t_hash = NFIX_AlignmentLogic.get_texture_hash(t_obj)
            t_pts = NFIX_AlignmentLogic.get_evaluated_world_points(t_obj, depsgraph)
            if t_pts.shape[0] > 0:
                GOLDEN_TARGETS.setdefault(t_hash, []).append(t_pts)

        if not any(c['name'] == "Capture 1" for c in CAPTURE_SETS):
            CAPTURE_SETS.insert(0, {
                'name': "Capture 1",
                'objects': first_set,
                'reference_mesh': st.source_obj.name
            })
            context.scene["ninjafix_capture_sets"] = json.dumps([
                {'name': c['name'], 'objects': list(c['objects']), 'reference_mesh': c.get('reference_mesh','')}
                for c in CAPTURE_SETS
            ])

        self.report({'INFO'}, "Initial alignment applied and target snapshot created.")
        return {'FINISHED'}

class NFIX_OT_ForceCapture1(Operator):
    bl_idname = "nfix.force_capture1"
    bl_label = "Force Capture 1"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global CAPTURE_SETS, GOLDEN_TARGETS
        scene = context.scene

        # Record all current Ninja meshes as Capture 1
        first_set = current_ninja_set(scene)
        CAPTURE_SETS.insert(0, {
            'name': "Capture 1",
            'objects': first_set,
            'reference_mesh': ''
        })
        scene["ninjafix_capture_sets"] = json.dumps([
            {'name': c['name'], 'objects': list(c['objects']), 'reference_mesh': c.get('reference_mesh','')}
            for c in CAPTURE_SETS
        ])

        # Build in-memory blueprint (golden targets) from entire scene
        GOLDEN_TARGETS.clear()
        depsgraph = context.evaluated_depsgraph_get()
        for obj in scene.objects:
            if obj.type != 'MESH':
                continue
            t_hash = NFIX_AlignmentLogic.get_texture_hash(obj)
            t_pts = NFIX_AlignmentLogic.get_evaluated_world_points(obj, depsgraph)
            if t_pts.shape[0] > 0:
                GOLDEN_TARGETS.setdefault(t_hash, []).append(t_pts)

        self.report({'INFO'}, "Force Capture 1 created and blueprint cached.")
        return {'FINISHED'}

# =================================================================================================
# STAGE 4: PROCESS ALL OTHER CAPTURES
# =================================================================================================
class NFIX_OT_ProcessAllCaptures(Operator):
    bl_idname      = "nfix.process_all_captures"
    bl_label       = "Process All Captures"
    bl_options     = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return NUMPY_INSTALLED and len(CAPTURE_SETS) > 1 and bool(GOLDEN_TARGETS)

    def execute(self, context):
        global INITIAL_ALIGN_MATRIX, GOLDEN_TARGETS, PREFERRED_ANCHOR_HASHES

        if not GOLDEN_TARGETS:
            self.report({'ERROR'}, "No cached target data found. Run Stage 2 first.")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        tol = 0.05  # max allowed vertex distance
        aligned_count = 0
        skipped = 0

        for cap in CAPTURE_SETS[1:]:
            # Get all ninja objects for the current capture
            all_ninjas_in_cap = [
                context.scene.objects.get(n)
                for n in cap['objects']
                if is_ninja(context.scene.objects.get(n))
            ]

            # NEW: Prioritize ninjas based on PREFERRED_ANCHOR_HASHES
            preferred_ninjas = []
            other_ninjas = []
            for o in all_ninjas_in_cap:
                if o is None: continue
                h = NFIX_AlignmentLogic.get_texture_hash(o)
                # Check if hash is valid and in the preferred set
                if h and h in PREFERRED_ANCHOR_HASHES:
                    preferred_ninjas.append(o)
                else:
                    other_ninjas.append(o)
            
            # Sort the remaining ninjas by vertex count as a fallback
            other_ninjas.sort(key=lambda o: len(o.data.vertices) if o else 0, reverse=True)

            # The final list to iterate over, with preferred anchors first
            ninjas_to_try = preferred_ninjas + other_ninjas
            
            matched = False
            for src in ninjas_to_try:
                if src is None: continue
                
                pts = NFIX_AlignmentLogic.get_evaluated_world_points(src, depsgraph)
                if pts.shape[0] == 0:
                    continue

                # skip perfectly symmetric shapes (two eigenvalues within 1%)
                cov = np.cov(pts.T)
                eig = np.linalg.eigvalsh(cov)
                eig.sort()
                if (eig[1] - eig[0]) / eig[2] < 0.01 or (eig[2] - eig[1]) / eig[2] < 0.01:
                    continue

                src_hash = NFIX_AlignmentLogic.get_texture_hash(src)
                if src_hash not in GOLDEN_TARGETS:
                    continue

                # test each cached target
                for tgt_pts in GOLDEN_TARGETS[src_hash]:
                    if pts.shape != tgt_pts.shape:
                        continue

                    Mtest = NFIX_AlignmentLogic.solve_alignment(pts, tgt_pts)
                    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
                    aligned = (pts_h @ np.array(Mtest).T)[:, :3]

                    # max per-vertex error
                    dists = np.linalg.norm(aligned - tgt_pts, axis=1)
                    if dists.max() < tol:
                        # Successful alignment, apply the transform to all objects in the capture
                        for nm in cap['objects']:
                            ob = context.scene.objects.get(nm)
                            if is_ninja(ob):
                                if "ninjafix_prev_matrix" not in ob:
                                    ob["ninjafix_prev_matrix"] = mat_to_list(ob.matrix_world)
                                
                                ob.matrix_world = Mtest @ ob.matrix_world
                        
                        # NEW: Add the successful anchor hash to the preferred set for next captures
                        successful_hash = NFIX_AlignmentLogic.get_texture_hash(src)
                        if successful_hash:
                            PREFERRED_ANCHOR_HASHES.add(successful_hash)

                        cap['reference_mesh'] = src.name
                        self.report({'INFO'},
                            f"'{cap['name']}': aligned via '{src.name}' (max-dist={dists.max():.4f})")
                        aligned_count += 1
                        matched = True
                        break # Exit the inner loop (tgt_pts)
                
                if matched:
                    break # Exit the outer loop (src)

            if not matched:
                self.report({'WARNING'}, f"'{cap['name']}': no suitable non-symmetrical anchor found, skipped.")
                skipped += 1

        # update scene property
        context.scene["ninjafix_capture_sets"] = json.dumps([
            {'name': c['name'], 'objects': list(c['objects']), 'reference_mesh': c.get('reference_mesh','')}
            for c in CAPTURE_SETS
        ])

        self.report({'INFO'},
            f"Processing complete: {aligned_count} aligned, {skipped} skipped.")
        return {'FINISHED'}

# =================================================================================================
# Other Operators & UI
# =================================================================================================
# (These classes remain mostly unchanged, but are included for a complete script)

class NFIX_OT_UndoLastTransform(Operator):
    bl_idname = "nfix.undo_last_transform"
    bl_label = "Undo Last Transform"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        restored = 0
        for ob in context.scene.objects:
            if "ninjafix_prev_matrix" in ob:
                vals = ob["ninjafix_prev_matrix"]
                ob.matrix_world = Matrix([vals[i*4:(i+1)*4] for i in range(4)])
                del ob["ninjafix_prev_matrix"]
                restored += 1
        self.report({'INFO'}, f"Restored {restored} Ninja mesh transforms.")
        return {'FINISHED'}

class NFIX_OT_SetCurrentCapture(Operator):
    bl_idname = "nfix.set_current_capture"
    bl_label = "Add New Capture"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ninjas = current_ninja_set(context.scene)
        recorded = set().union(*(c['objects'] for c in CAPTURE_SETS))
        unrec = ninjas - recorded
        if not unrec:
            self.report({'WARNING'}, "No new Ninja meshes found to add as a capture.")
            return {'CANCELLED'}
        name = f"Capture {len(CAPTURE_SETS) + 1}"
        CAPTURE_SETS.append({'name': name, 'objects': unrec, 'reference_mesh': ''})
        context.scene["ninjafix_capture_sets"] = json.dumps([{'name': c['name'], 'objects': list(c['objects']), 'reference_mesh': c.get('reference_mesh','')} for c in CAPTURE_SETS])
        self.report({'INFO'}, f"Added {name} with {len(unrec)} meshes.")
        return {'FINISHED'}

# 1) PropertyGroup for each folder entry
class NFIX_FolderItem(PropertyGroup):
    name: StringProperty()
    path: StringProperty()
    use: BoolProperty(default=False)

# 3) Operator to scan subfolders
class NFIX_OT_ScanCaptureFolders(Operator):
    bl_idname = "nfix.scan_capture_folders"
    bl_label = "Scan Subfolders"
    bl_description = "Scan the selected parent directory for subfolders (or .nr files) to import"

    def execute(self, context):
        scene = context.scene
        parent = scene.nfix_parent_dir
        scene.nfix_folder_items.clear()
        if not parent or not os.path.isdir(parent):
            self.report({'ERROR'}, "Please set a valid parent directory.")
            return {'CANCELLED'}
        try:
            entries = os.listdir(parent)
        except Exception as e:
            self.report({'ERROR'}, f"Cannot list directory: {e}")
            return {'CANCELLED'}
        # First, add each subfolder
        subfolder_count = 0
        for name in sorted(entries):
            full = os.path.join(parent, name)
            if os.path.isdir(full):
                item = scene.nfix_folder_items.add()
                item.name = name
                item.path = full
                item.use = False
                subfolder_count += 1
        # If no subfolders found but .nr files exist directly, add a special "parent files" entry
        if subfolder_count == 0:
            nr_files = [f for f in entries if f.lower().endswith('.nr')]
            if nr_files:
                item = scene.nfix_folder_items.add()
                item.name = "<Parent .nr Files>"
                item.path = parent
                item.use = False
                self.report({'INFO'}, f"Found {len(nr_files)} .nr files in parent folder.")
        if subfolder_count == 0 and not scene.nfix_folder_items:
            self.report({'WARNING'}, "No subfolders or .nr files found.")
        else:
            if subfolder_count > 0:
                self.report({'INFO'}, f"Found {subfolder_count} subfolders.")
        return {'FINISHED'}

# 4) UIList to display folders
class NFIX_UL_FolderList(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # data is the CollectionProperty owner, item is NFIX_FolderItem
        split = layout.split(factor=0.1)
        split.prop(item, "use", text="")
        split.label(text=item.name, translate=False, icon='FILE_FOLDER')

# 5) Operator to import selected folders
class NFIX_OT_ImportSelectedFolders(Operator):
    bl_idname = "nfix.import_selected_folders"
    bl_label = "Import Selected Folders"
    bl_description = "Import Ninja meshes from the checked subfolders"

    @classmethod
    def poll(cls, context):
        st = context.scene.ninjafix_settings
        return st.import_folders and bool(context.scene.nfix_folder_items)

    def execute(self, context):
        scene = context.scene
        st = scene.ninjafix_settings
        selected = [item for item in scene.nfix_folder_items if item.use]
        if not selected:
            self.report({'WARNING'}, "No folders selected.")
            return {'CANCELLED'}

        if not hasattr(bpy.ops.import_mesh, "nr"):
            self.report({'ERROR'}, "Ninja Ripper importer (import_mesh.nr) not found.")
            return {'CANCELLED'}

        imported = 0
        for item in selected:
            folder = item.path
            if not os.path.isdir(folder):
                self.report({'WARNING'}, f"Not a directory: {item.name}")
                continue

            # capture current meshes, import batch, then diff once per folder
            before = current_ninja_set(scene)
            nr_files = [f for f in os.listdir(folder) if f.lower().endswith(".nr")]
            if not nr_files:
                self.report({'WARNING'}, f"No .nr files in '{item.name}'.")
                continue

            files_param = [{"name": fn} for fn in nr_files]
            bpy.ops.import_mesh.nr(
                directory=folder,
                files=files_param,
                loadExtraUvData=True,
                projFov_useRH=st.flip_geometry
            )

            after = current_ninja_set(scene)
            new_meshes = after - before
            if new_meshes:
                imported += 1
                capture_name = f"Capture {len(CAPTURE_SETS) + 1}"
                CAPTURE_SETS.append({
                    "name": capture_name,
                    "objects": new_meshes,
                    "reference_mesh": ""
                })
                self.report(
                    {'INFO'},
                    f"Imported {len(new_meshes)} meshes from '{item.name}' as {capture_name}."
                )

        if imported:
            scene["ninjafix_capture_sets"] = json.dumps([
                {
                    "name": cap["name"],
                    "objects": list(cap["objects"]),
                    "reference_mesh": cap.get("reference_mesh", "")
                }
                for cap in CAPTURE_SETS
            ])
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    for region in area.regions:
                        if region.type == "UI":
                            region.tag_redraw()

        return {'FINISHED'}

class NFIX_OT_RemoveCapture(Operator):
    bl_idname = "nfix.remove_capture"
    bl_label = "X"
    bl_options = {'REGISTER', 'UNDO'}
    capture_name: StringProperty()
    def execute(self, context):
        global CAPTURE_SETS
        CAPTURE_SETS = [c for c in CAPTURE_SETS if c['name'] != self.capture_name]
        context.scene["ninjafix_capture_sets"] = json.dumps([{'name': c['name'], 'objects': list(c['objects']), 'reference_mesh': c.get('reference_mesh','')} for c in CAPTURE_SETS])
        return {'FINISHED'}

class VIEW3D_PT_NinjaFix_Aligner(Panel):
    bl_label = "NinjaFix Aligner"
    bl_idname = "VIEW3D_PT_nfix_aligner"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "NinjaFix"

    def draw(self, context):
        import bpy, json
        layout = self.layout
        st = context.scene.ninjafix_settings

        # Determine fallback for the "ROTATE" icon if not available
        icon_items = bpy.types.UILayout.bl_rna.functions['operator'].parameters['icon'].enum_items.keys()
        rotate_icon = 'ROTATE' if 'ROTATE' in icon_items else 'FILE_TICK'

        # Stage 1: Select Meshes
        if not NUMPY_INSTALLED:
            layout.operator('nfix.install_numpy', icon='ERROR')
            layout.label(text="Restart Blender after installation.")
            return

        box = layout.box()
        box.label(text="Stage 1: Select Meshes", icon='OBJECT_DATA')
        box.operator('nfix.set_objects', icon='CHECKMARK')
        if st.source_obj:
            box.label(text=f"Ninja: {st.source_obj.name}", icon='MOD_MESHDEFORM')
        if st.target_obj:
            box.label(text=f"Remix: {st.target_obj.name}", icon='MESH_PLANE')

        # Stage 2: Initial Alignment
        box = layout.box()
        box.label(text="Stage 2: Initial Alignment", icon='PLAY')
        box.enabled = bool(st.source_obj and st.target_obj)
        box.operator(
            'nfix.calculate_and_batch_fix',
            text="Generate & Apply Fix",
            icon=rotate_icon
        )

        # Stage 3: Manage Captures
        box = layout.box()
        box.label(text="Stage 3: Manage Captures", icon='BOOKMARKS')
        box.prop(st, "auto_capture")

        # Import From Folders toggle + UI
        box.prop(st, "import_folders", text="Import From Folders", icon='FILE_FOLDER')
        if st.import_folders:
            box.prop(st, "flip_geometry", text="Flip Geometry")
            box.prop(context.scene, "nfix_parent_dir", text="Parent Folder")
            box.operator("nfix.scan_capture_folders", icon='FILE_FOLDER')
            box.template_list(
                "NFIX_UL_FolderList", "",
                context.scene, "nfix_folder_items",
                context.scene, "nfix_folder_index",
                rows=4
            )
            box.operator("nfix.import_selected_folders", icon='IMPORT')

        # Existing capture sets listing
        if not CAPTURE_SETS and "ninjafix_capture_sets" in context.scene:
            try:
                CAPTURE_SETS[:] = json.loads(context.scene["ninjafix_capture_sets"])
            except:
                pass

        for cap in CAPTURE_SETS:
            row = box.row(align=True)
            ref = cap.get('reference_mesh', '') or '—'
            icon = 'CHECKMARK' if cap.get('reference_mesh') else 'QUESTION'
            row.label(text=f"{cap['name']}: {ref}", icon=icon)
            op = row.operator('nfix.remove_capture', text="", icon='X')
            op.capture_name = cap['name']

        if not CAPTURE_SETS:
            box.operator('nfix.force_capture1', icon='BOOKMARKS', text="Force Capture 1")
        else:
            box.operator('nfix.set_current_capture', text="Add New Capture", icon='ADD')

        # Stage 4: Process All & Cleanup
        box = layout.box()
        box.label(text="Stage 4: Process All & Cleanup", icon='FILE_TICK')
        box.operator('nfix.process_all_captures', text="Process All Captures", icon='CHECKMARK')
        box.operator('nfix.undo_last_transform', text="Undo Last Transform", icon='LOOP_BACK')
        box.operator('nfix.remove_mat_duplicates', text="Remove mat_ Duplicates", icon='TRASH')

# =================================================================================================
# Registration
# =================================================================================================
classes = (
    NinjaFixSettings,
    NFIX_FolderItem,
    NFIX_OT_InstallNumpy,
    NFIX_OT_SetObjects,
    NFIX_OT_ScanCaptureFolders,
    NFIX_UL_FolderList,
    NFIX_OT_ImportSelectedFolders,
    NFIX_OT_CalculateAndBatchFix,
    NFIX_OT_ForceCapture1,
    NFIX_OT_ProcessAllCaptures,
    NFIX_OT_UndoLastTransform,
    NFIX_OT_SetCurrentCapture,
    NFIX_OT_RemoveCapture,
    NFIX_OT_RemoveMatDuplicates,
    NFIX_OT_AutoCaptureTimer,
    VIEW3D_PT_NinjaFix_Aligner,
)

# =================================================================================================
# Registration (with reload of saved matrix)
# =================================================================================================
def register():
    # 1) register all classes
    for cls in classes:
        bpy.utils.register_class(cls)

    # 2) define the new Scene-level props for the folder UI
    bpy.types.Scene.nfix_parent_dir = StringProperty(
        name="Parent Folder",
        description="Select the parent directory containing capture subfolders",
        subtype='DIR_PATH'
    )
    bpy.types.Scene.nfix_folder_items = CollectionProperty(type=NFIX_FolderItem)
    bpy.types.Scene.nfix_folder_index = IntProperty()

    # 3) your existing NinjaFixSettings registration
    bpy.types.Scene.ninjafix_settings = PointerProperty(type=NinjaFixSettings)

    # === Restore saved alignment matrix (only if a .blend is open) ===
    try:
        blend_file = bpy.data.filepath
        # only proceed if we actually have a blend path
        if blend_file:
            db_entry = load_db().get(blend_file, {})
            mat_list = db_entry.get("matrix")
            if mat_list:
                from mathutils import Matrix
                global INITIAL_ALIGN_MATRIX
                INITIAL_ALIGN_MATRIX = Matrix([
                    mat_list[i*4:(i+1)*4] for i in range(4)
                ])
    except Exception:
        # bpy.data may be restricted during addon enable; ignore in that case
        pass

def unregister():
    # remove our Scene props
    del bpy.types.Scene.nfix_parent_dir
    del bpy.types.Scene.nfix_folder_items
    del bpy.types.Scene.nfix_folder_index

    # remove the old settings pointer
    del bpy.types.Scene.ninjafix_settings

    # unregister classes in reverse
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
