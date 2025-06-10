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
from bpy.props import PointerProperty, StringProperty
from bpy.types import PropertyGroup, Operator, Panel
import json
from mathutils import Vector, Matrix, kdtree
import hashlib
import os
import numpy as np

# --- Globals ---
CAPTURE_SETS = []
INITIAL_ALIGN_MATRIX = None
GOLDEN_TARGETS = {} # NEW: To store the 'snapshot' of target points

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
        with caching of evaluated points, KD-trees, and material-hash checks.
        """
        import numpy as np
        from mathutils import Matrix, kdtree
        import re

        depsgraph = context.evaluated_depsgraph_get()
        scene = context.scene
        tol = 0.005

        print("\n================ NinjaFix DEBUG START ================")
        print(f"[DEBUG] Tolerance for duplicate detection: {tol}")

        # ---------------------------------------------------------------------
        # 1) PRECOMPUTE & CACHE: world-space points, KD-trees, mat-hash flags
        # ---------------------------------------------------------------------
        world_pts_cache = {}
        kd_cache        = {}
        matflag_cache   = {}
        entries         = []
        hash_pat        = re.compile(r"^mat_[A-F0-9]{6,}$", re.I)

        def has_normal_mat(obj):
            if obj.name in matflag_cache:
                return matflag_cache[obj.name]
            # determine if object has a "normal" material
            for slot in obj.material_slots:
                mat = slot.material
                if mat is None or not hash_pat.fullmatch(mat.name):
                    matflag_cache[obj.name] = True
                    return True
            matflag_cache[obj.name] = not obj.material_slots
            return matflag_cache[obj.name]

        # collect meshes
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

            entries.append({
                'obj': ob,
                'pts': pts,
                'vcount': vcount,
                'kd': kd,
                'has_normal': has_normal_mat(ob),
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
                        print(f"[DEBUG] Marking duplicates (tol={tol:.6f})")

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
        # ---------------------------------------------------------------------
        to_delete = []
        for ci, comp in enumerate(clusters):
            if len(comp) < 2:
                continue
            ents = [entries[k] for k in comp]
            names = [e['obj'].name for e in ents]
            print(f"\n[DEBUG] Processing Cluster {ci}: {names}")

            keeper = None
            reason = ""
            # 1) pure Ninja mesh (name starts with mesh_ and no dot)
            for e in ents:
                nm = e['obj'].name
                if nm.startswith("mesh_") and "." not in nm:
                    keeper = e['obj']
                    reason = "pure Ninja"
                    break
            # 2) has normal material
            if keeper is None:
                for e in ents:
                    if e['has_normal']:
                        keeper = e['obj']
                        reason = "has normal material"
                        break
            # 3) fallback
            if keeper is None:
                keeper = ents[0]['obj']
                reason = "first fallback"
            print(f"[DEBUG] Keeper: '{keeper.name}' ({reason})")

            for e in ents:
                ob_e = e['obj']
                if ob_e is not keeper:
                    print(f"[DEBUG] Marking '{ob_e.name}' for deletion")
                    to_delete.append(ob_e)

        # ---------------------------------------------------------------------
        # 6) BATCH DELETE SAFELY
        # ---------------------------------------------------------------------
        deleted = 0
        print(f"\n[DEBUG] Deleting {len(to_delete)} objects:")
        # We do all unlinks in one pass before removes to avoid modifying scene.objects mid-loop
        for ob in to_delete:
            for coll in list(ob.users_collection):
                coll.objects.unlink(ob)
        for ob in to_delete:
            name = ob.name
            print(f"[DEBUG] Deleting '{name}'")
            if ob.type == 'MESH':
                mat = ob.matrix_world.copy()
                try:
                    ob.data.transform(mat)
                except Exception as e:
                    print(f"[DEBUG] Warn: bake failed for '{name}': {e}")
                ob.matrix_world = Matrix.Identity(4)
            try:
                bpy.data.objects.remove(ob, do_unlink=True)
                deleted += 1
                print(f"[NinjaFix] Deleted '{name}'")
            except Exception as e:
                print(f"[DEBUG] Warn: removal failed for '{name}': {e}")
        self.report({'INFO'}, f"Removed {deleted} duplicates.")
        print("================ NinjaFix DEBUG END ================\n")
        return {'FINISHED'}

# =================================================================================================
# Property Groups & UI Setup
# =================================================================================================
class NinjaFixSettings(PropertyGroup):
    source_obj: PointerProperty(type=bpy.types.Object)
    target_obj: PointerProperty(type=bpy.types.Object)

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
class NFIX_OT_CalculateAndBatchFix(Operator):
    bl_idname = "nfix.calculate_and_batch_fix"
    bl_label  = "Generate & Apply Fix"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        st = context.scene.ninjafix_settings
        return NUMPY_INSTALLED and st.source_obj and st.target_obj

    def execute(self, context):
        global INITIAL_ALIGN_MATRIX, GOLDEN_TARGETS

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
        global INITIAL_ALIGN_MATRIX, GOLDEN_TARGETS

        if not GOLDEN_TARGETS:
            self.report({'ERROR'}, "No cached target data found. Run Stage 2 first.")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        tol = 0.05  # max allowed vertex distance
        aligned_count = 0
        skipped = 0

        for cap in CAPTURE_SETS[1:]:
            ninjas = [
                context.scene.objects.get(n)
                for n in cap['objects']
                if is_ninja(context.scene.objects.get(n))
            ]
            # try largest mesh first
            ninjas.sort(key=lambda o: len(o.data.vertices) if o else 0, reverse=True)

            matched = False
            for src in ninjas:
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
                        # FIXED: Apply the complete transformation directly
                        # instead of compounding with existing matrix
                        for nm in cap['objects']:
                            ob = context.scene.objects.get(nm)
                            if is_ninja(ob):
                                if "ninjafix_prev_matrix" not in ob:
                                    ob["ninjafix_prev_matrix"] = mat_to_list(ob.matrix_world)
                                
                                # CORRECTED: Apply the full transformation matrix directly
                                # instead of Mtest @ current matrix
                                ob.matrix_world = Mtest
                                
                                # Optional: Apply non-uniform scale if needed
                                # loc, rot, scale = Mtest.decompose()
                                # ob.scale = scale
                                
                        cap['reference_mesh'] = src.name
                        self.report({'INFO'},
                            f"'{cap['name']}': aligned via '{src.name}' (max-dist={dists.max():.4f})")
                        aligned_count += 1
                        matched = True
                        break

                if matched:
                    break

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
    bl_label       = "NinjaFix Aligner"
    bl_idname      = "VIEW3D_PT_nfix_aligner"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "NinjaFix"

    def draw(self, context):
        layout = self.layout
        st = context.scene.ninjafix_settings

        if not NUMPY_INSTALLED:
            layout.operator('nfix.install_numpy', icon='ERROR')
            layout.label(text="Restart Blender after installation.")
            return

        # Stage 1: Select Meshes
        box = layout.box()
        box.label(text="Stage 1: Select Meshes", icon='OBJECT_DATA')
        col = box.column()
        col.operator('nfix.set_objects', icon='CHECKMARK')
        if st.source_obj:
            box.label(text=f"Ninja: {st.source_obj.name}", icon='MOD_MESHDEFORM')
        if st.target_obj:
            box.label(text=f"Remix: {st.target_obj.name}", icon='MESH_PLANE')

        # Stage 2: Initial Alignment
        box = layout.box()
        box.label(text="Stage 2: Initial Alignment", icon='PLAY')
        box.enabled = bool(st.source_obj and st.target_obj)
        box.operator('nfix.calculate_and_batch_fix', text="Generate & Apply Fix", icon='ROTATE')

        # Stage 3: Manage Captures
        box = layout.box()
        box.label(text="Stage 3: Manage Captures", icon='BOOKMARKS')
        # reload from scene prop if needed
        if not CAPTURE_SETS and "ninjafix_capture_sets" in context.scene:
            try:
                CAPTURE_SETS[:] = json.loads(context.scene["ninjafix_capture_sets"])
            except:
                pass

        for cap in CAPTURE_SETS:
            row = box.row(align=True)
            ref = cap.get('reference_mesh', '') or '—'
            row.label(
                text=f"{cap['name']}: {ref}",
                icon='CHECKMARK' if cap.get('reference_mesh') else 'QUESTION'
            )
            op = row.operator('nfix.remove_capture', text="", icon='X')
            op.capture_name = cap['name']

        # If we have no captures yet, show Force Capture 1
        if not CAPTURE_SETS:
            box.operator('nfix.force_capture1', icon='BOOKMARKS', text="Force Capture 1")
        else:
            box.operator('nfix.set_current_capture', text="Add New Capture", icon='ADD')

        # Stage 4: Process All & Cleanup
        box = layout.box()
        box.label(text="Stage 4: Process All", icon='FILE_TICK')
        box.operator('nfix.process_all_captures', text="Process All Captures", icon='CHECKMARK')
        box.operator('nfix.undo_last_transform', text="Undo Last Transform", icon='LOOP_BACK')
        # NEW button to delete duplicates using mat_ rule
        box.operator('nfix.remove_mat_duplicates', text="Remove mat_ Duplicates", icon='TRASH')

# =================================================================================================
# Registration
# =================================================================================================
classes = (
    NinjaFixSettings,
    NFIX_OT_InstallNumpy,
    NFIX_OT_SetObjects,
    NFIX_OT_CalculateAndBatchFix,
    NFIX_OT_ForceCapture1,
    NFIX_OT_ProcessAllCaptures,
    NFIX_OT_UndoLastTransform,
    NFIX_OT_SetCurrentCapture,
    NFIX_OT_RemoveCapture,
    NFIX_OT_RemoveMatDuplicates,    # ← add this
    VIEW3D_PT_NinjaFix_Aligner,
)

# =================================================================================================
# Registration (with reload of saved matrix)
# =================================================================================================
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
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
    del bpy.types.Scene.ninjafix_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
