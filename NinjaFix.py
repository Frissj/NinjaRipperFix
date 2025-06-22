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
from collections import Counter

# --- Globals ---
CAPTURE_SETS = []
INITIAL_ALIGN_MATRIX = None
# OLD DBs are replaced by new, optimized versions
GOLDEN_TARGETS_BY_VCOUNT = {} # NEW: {vcount: {hash: world_pts}}
GOLDEN_GEOMETRY_DB_BY_VCOUNT = {} # NEW: {vcount: {hash: local_pts}}
PREFERRED_ANCHOR_GEOM_HASHES = set()
NON_UNIQUE_GEOM_HASHES = set()
SEEN_GEOM_HASHES = set()
TEXTURE_PIXEL_CACHE = {}
_TEXTURE_HASH_CACHE = {}

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

def get_geometry_hash(obj):
    """
    Creates a SHA256 hash from an object's LOCAL vertex coordinates.
    
    FINAL VERSION: This version uses quantization to handle floating-point noise
    AND canonical sorting to handle differences in vertex order. This provides
    the most robust possible fingerprint for identifying identical base geometry.
    """
    if not obj or obj.type != 'MESH' or not obj.data.vertices:
        return ""
    
    v_count = len(obj.data.vertices)
    local_coords = np.empty(v_count * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", local_coords)
    local_coords.shape = (v_count, 3)
    
    # 1. Quantize coordinates to remove floating-point noise.
    quantization_factor = 10000.0
    quantized_coords = np.round(local_coords * quantization_factor).astype(np.int64)
    
    # 2. Sort vertices in a canonical order (by X, then Y, then Z) to be
    # immune to vertex order differences. We must use a structured array for this.
    structured_array = np.core.records.fromarrays(quantized_coords.transpose(), names='x, y, z')
    structured_array.sort(order=['x', 'y', 'z'], axis=0)

    # 3. The hash is now based on the bytes of this stable, sorted, integer array.
    return hashlib.sha256(structured_array.tobytes()).hexdigest()

def calculate_shape_difference(pts1, pts2):
    """
    Calculates the Root Mean Square Deviation (RMSD) between two point clouds
    after aligning them via rotation and translation. This is a measure of
    pure shape difference. Returns a single float value.
    """
    v1 = pts1.shape[0]
    v2 = pts2.shape[0]

    if v1 != v2 or v1 < 3:
        return -1.0 # Return -1 to indicate an error/mismatch

    # Use float64 for precision in geometric calculations
    pts1 = pts1.astype(np.float64)
    pts2 = pts2.astype(np.float64)

    # 1. Center both point clouds
    centroid1 = np.mean(pts1, axis=0)
    centroid2 = np.mean(pts2, axis=0)
    pts1_centered = pts1 - centroid1
    pts2_centered = pts2 - centroid2

    # 2. Compute the cross-covariance matrix H
    H = pts1_centered.T @ pts2_centered

    # 3. Find the optimal rotation using Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correct for reflection if necessary
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = Vt.T @ U.T

    # 4. Apply the optimal rotation
    pts1_aligned = pts1_centered @ R

    # 5. Calculate and return the final RMSD
    diff = pts1_aligned - pts2_centered
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd

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
            temp_mesh = eval_obj.to_mesh()
            
            if not temp_mesh.vertices:
                return np.array([])

            size = len(temp_mesh.vertices)
            points = np.empty(size * 3, dtype=np.float32)
            temp_mesh.vertices.foreach_get("co", points)
            points.shape = (size, 3)
            
            matrix = np.array(eval_obj.matrix_world)
            points_h = np.hstack([points, np.ones((size, 1))])
            world_points = (points_h @ matrix.T)[:, :3]

            return world_points

        finally:
            if 'temp_mesh' in locals() and temp_mesh:
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
        M_T_transposed = M_T.T
        for r in range(3):
            for c in range(4):
                M[r][c] = M_T_transposed[r, c]

        return M

class NFIX_OT_DebugShapeCompare(Operator):
    """
    The primary diagnostic tool.
    - 1 mesh: Full Anchor Candidate Report.
    - 2 meshes: Precise shape difference (RMSD) analysis.
    - 4 meshes (2 Ninja, 2 Remix): Full anchor pair validation.
    """
    bl_idname = "nfix.debug_shape_compare"
    bl_label = "Generate Anchor Report"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (
            len(context.selected_objects) in (1, 2, 4)
            and all(o.type == 'MESH' for o in context.selected_objects)
        )

    def execute(self, context):
        import numpy as np
        from mathutils import Matrix
        import os

        selected = context.selected_objects
        depsgraph = context.evaluated_depsgraph_get()

        # --- 1 object: Anchor Candidate Report ---
        if len(selected) == 1:
            # This mode remains as is for quick individual checks
            obj = selected[0]
            report_lines = [
                "\n" + "="*50,
                " NINJAFIX ANCHOR CANDIDATE REPORT (GEOMETRY-ONLY)",
                "="*50,
                f"Analyzing Candidate: {obj.name}\n"
            ]
            report_lines.append("--- 1. Uniqueness ---")
            if not NON_UNIQUE_GEOM_HASHES and GOLDEN_GEOMETRY_DB:
                print("Warning: Uniqueness database not yet built (requires running Stage 4 once).")
            gh = get_geometry_hash(obj)
            unique = gh not in NON_UNIQUE_GEOM_HASHES
            report_lines.append(f"Result: {'UNIQUE' if unique else 'NON-UNIQUE'}")
            report_lines.append(f"Verdict: {'PASSED.' if unique else 'INVALID ANCHOR.'}")
            report_lines.append("")
            report_lines.append("--- 2. Golden Counterpart ---")
            if not GOLDEN_GEOMETRY_DB:
                 report_lines.append("Result: SKIPPED (Golden Database not yet built - run Stage 2)")
            else:
                v_count = len(obj.data.vertices)
                local_coords = np.empty(v_count * 3, dtype=np.float32)
                obj.data.vertices.foreach_get("co", local_coords)
                local_coords.shape = (v_count, 3)
                match_hash, rmsd = find_best_geom_match(local_coords, GOLDEN_GEOMETRY_DB)
                has_gold = match_hash is not None
                report_lines.append(f"Result: {'Found' if has_gold else 'NOT Found'} (Best RMSD: {rmsd:.8f})")
                report_lines.append(f"Verdict: {'PASSED.' if has_gold else 'INVALID ANCHOR.'}")
                report_lines.append("")
            final = "POOR"
            if 'unique' in locals() and unique and 'has_gold' in locals() and has_gold:
                final = "GOOD"
            summary = f"This is a {final} anchor candidate."
            report_lines += ["\n" + "="*50, f"  FINAL VERDICT: {summary}", "="*50]
            for line in report_lines:
                print(line)
            self.report({'INFO'}, f"Report Generated. Final Verdict: {summary}")
            return {'FINISHED'}

        # --- 2 objects: Precise Shape Comparison ---
        elif len(selected) == 2:
            # This mode remains as is for quick individual checks
            a, b = selected
            if a.data.vertices and b.data.vertices:
                a_local_pts = np.empty(len(a.data.vertices)*3, 'f'); a.data.vertices.foreach_get('co', a_local_pts); a_local_pts.shape=(-1,3)
                b_local_pts = np.empty(len(b.data.vertices)*3, 'f'); b.data.vertices.foreach_get('co', b_local_pts); b_local_pts.shape=(-1,3)
                rmsd = calculate_shape_difference(a_local_pts, b_local_pts) if a_local_pts.shape == b_local_pts.shape else -1.0
            else:
                rmsd = -1.0
            print("\n" + "="*50)
            print(" NINJAFIX SHAPE SIMILARITY ANALYSIS")
            print("="*50)
            print(f"Object 1: {a.name}")
            print(f"Object 2: {b.name}\n")
            if rmsd >= 0:
                print(f"Intrinsic Shape Difference (RMSD): {rmsd:.8f}\n")
                conclusion = "IDENTICAL SHAPE" if rmsd < 1e-6 else f"SIMILAR SHAPE (RMSD: {rmsd:.8f})"
            else:
                conclusion = "ERROR: Vertex counts do not match."
            print(f"Conclusion: {conclusion}")
            print("="*50)
            self.report({'INFO'}, f"Analysis Complete. Summary: {conclusion}")
            return {'FINISHED'}

        # --- REBUILT 4-object Anchor Pair Validation ---
        elif len(selected) == 4:
            ninjas = [o for o in selected if is_ninja(o)]
            remixes = [o for o in selected if is_remix(o)]
            
            if len(ninjas) != 2 or len(remixes) != 2:
                self.report({'ERROR'}, "Selection must contain exactly 2 Ninja and 2 Remix meshes.")
                return {'CANCELLED'}

            # This helper function performs the full test for one possible pairing configuration
            def test_pairing(n_a, n_b, r_a, r_b, depsgraph):
                report = []
                report.append(f"\n--- Testing Pairing: ('{n_a.name}' -> '{r_a.name}') AND ('{n_b.name}' -> '{r_b.name}') ---")
                
                n_a_w = NFIX_AlignmentLogic.get_evaluated_world_points(n_a, depsgraph)
                n_b_w = NFIX_AlignmentLogic.get_evaluated_world_points(n_b, depsgraph)
                r_a_w = NFIX_AlignmentLogic.get_evaluated_world_points(r_a, depsgraph)
                r_b_w = NFIX_AlignmentLogic.get_evaluated_world_points(r_b, depsgraph)

                if n_a_w.shape[0] < 4 or n_a_w.shape != r_a_w.shape or n_b_w.shape[0] < 4 or n_b_w.shape != r_b_w.shape:
                    report.append("Result: FAIL - Mismatched vertex counts in at least one pair.")
                    return report, "INVALID"

                # 1. Solve for the transforms
                M_a = NFIX_AlignmentLogic.solve_alignment(n_a_w, r_a_w)
                M_b = NFIX_AlignmentLogic.solve_alignment(n_b_w, r_b_w)

                # 2. Check Alignment Quality (Post-Transform RMSD)
                # This checks how well the calculated transform actually aligns the points.
                shape_match_tol = 0.07 # Same as main script
                
                n_a_aligned = (np.hstack([n_a_w, np.ones((n_a_w.shape[0],1))]) @ np.array(M_a).T)[:,:3]
                rmsd_a = float(np.sqrt(np.mean(np.sum((n_a_aligned - r_a_w)**2, axis=1))))
                align_a_ok = rmsd_a < shape_match_tol

                n_b_aligned = (np.hstack([n_b_w, np.ones((n_b_w.shape[0],1))]) @ np.array(M_b).T)[:,:3]
                rmsd_b = float(np.sqrt(np.mean(np.sum((n_b_aligned - r_b_w)**2, axis=1))))
                align_b_ok = rmsd_b < shape_match_tol
                
                report.append(f"  Alignment Quality 1: RMSD={rmsd_a:.8f} -> {'OK' if align_a_ok else 'FAIL'}")
                report.append(f"  Alignment Quality 2: RMSD={rmsd_b:.8f} -> {'OK' if align_b_ok else 'FAIL'}")

                # 3. Check Transform Consensus
                matrix_diff = np.linalg.norm(np.array(M_a) - np.array(M_b))
                matrix_tol = 0.01
                consensus_ok = matrix_diff < matrix_tol
                report.append(f"  Transform Consensus:  Diff={matrix_diff:.8f} -> {'OK' if consensus_ok else 'FAIL'}")

                # 4. Check Anchor Separation
                anchor_dist = np.linalg.norm(np.mean(n_a_w, axis=0) - np.mean(n_b_w, axis=0))
                dist_ok = anchor_dist > 1e-4
                report.append(f"  Anchor Separation:    Dist={anchor_dist:.4f} -> {'OK' if dist_ok else 'FAIL'}")
                
                # 5. Verdict for this pairing
                final_verdict = "VALID" if align_a_ok and align_b_ok and consensus_ok and dist_ok else "INVALID"
                report.append(f"  Verdict for this pairing: {final_verdict}")
                return report, final_verdict

            # --- Run the test for both possible configurations ---
            print("\n" + "="*60)
            print("      NINJAFIX ANCHOR PAIR VALIDATION REPORT")
            print("="*60)

            n1, n2 = ninjas
            r1, r2 = remixes
            
            # Configuration A: (n1,r1) and (n2,r2)
            report_a, verdict_a = test_pairing(n1, n2, r1, r2, depsgraph)
            for line in report_a:
                print(line)
            
            # Configuration B: (n1,r2) and (n2,r1)
            report_b, verdict_b = test_pairing(n1, n2, r2, r1, depsgraph)
            for line in report_b:
                print(line)
            
            print("\n" + "="*60)

            final_summary = "At least one VALID pairing found." if (verdict_a == "VALID" or verdict_b == "VALID") else "NO valid pairing found."
            self.report({'INFO'}, f"Validation Complete. {final_summary}")
            return {'FINISHED'}

        return {'CANCELLED'}

class NFIX_OT_SelectByTextureHash(Operator):
    """
    Selects all objects in the scene that share any Base Color
    texture with any of the currently selected objects.
    """
    bl_idname = "nfix.select_by_texture"
    bl_label = "Select Objects by Texture Hash"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # Enable the button if at least one object is selected.
        return len(context.selected_objects) > 0

    def execute(self, context):
        # 1. Get the list of initially selected mesh objects
        source_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not source_objects:
            self.report({'WARNING'}, "No mesh objects are selected.")
            return {'CANCELLED'}

        # 2. Collect all unique texture hashes from all selected source objects
        all_target_hashes = set()
        for obj in source_objects:
            # Your existing function finds all unique texture hashes on an object
            hashes = get_individual_texture_hashes(obj)
            all_target_hashes.update(hashes)

        if not all_target_hashes:
            self.report({'INFO'}, "Selected objects have no identifiable Base Color textures.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Found {len(all_target_hashes)} unique textures from {len(source_objects)} source objects.")

        # Store the active object to restore it later if it's part of the final selection
        last_active = context.active_object

        # 3. Deselect all objects in the scene
        bpy.ops.object.select_all(action='DESELECT')

        # 4. Iterate through all mesh objects in the scene and select matches
        final_selection = []
        for obj in context.scene.objects:
            if obj.type == 'MESH':
                obj_hashes = get_individual_texture_hashes(obj)
                # Select if the object's textures have any overlap with the target hashes
                if not all_target_hashes.isdisjoint(obj_hashes):
                    obj.select_set(True)
                    final_selection.append(obj)
        
        # 5. Restore the active object if it was part of the new selection
        if last_active in final_selection:
            context.view_layer.objects.active = last_active
        # Otherwise, make the first found object active
        elif final_selection:
            context.view_layer.objects.active = final_selection[0]

        self.report({'INFO'}, f"Selected {len(final_selection)} objects sharing common textures.")

        return {'FINISHED'}

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
        from mathutils import kdtree

        depsgraph = context.evaluated_depsgraph_get()
        scene = context.scene
        tol = 0.005

        # 1) COLLECT & CACHE DATA FOR EACH MESH
        entries = []                # will hold dicts with obj, points, KD-tree, flags
        matflag_cache = {}          # cache for has_normal_mat decisions

        def has_normal_mat(obj):
            """
            Returns True if the object has any “normal” material slot
            (i.e. no slots, an empty slot, or a material whose name does NOT start with 'mat_').
            """
            nm = obj.name
            if nm in matflag_cache:
                return matflag_cache[nm]
            # No materials → treat as normal
            if not obj.material_slots:
                matflag_cache[nm] = True
                return True
            # If any slot is empty or its material doesn’t start with 'mat_'
            for slot in obj.material_slots:
                mat = slot.material
                if mat is None or not mat.name.lower().startswith("mat_"):
                    matflag_cache[nm] = True
                    return True
            # Otherwise, it’s a mat_-only object
            matflag_cache[nm] = False
            return False

        # Gather world‐space vertices and build a KD‐tree for each mesh
        for ob in scene.objects:
            if ob.type != 'MESH':
                continue
            pts = NFIX_AlignmentLogic.get_evaluated_world_points(ob, depsgraph)
            if pts.shape[0] == 0:
                continue
            vcount = pts.shape[0]
            kd = kdtree.KDTree(vcount)
            for i, co in enumerate(pts):
                kd.insert(co, i)
            kd.balance()
            entries.append({
                'obj':       ob,
                'pts':       pts,
                'vcount':    vcount,
                'kd':        kd,
                'has_normal': has_normal_mat(ob),
            })

        # If fewer than two meshes, nothing to do
        if len(entries) < 2:
            self.report({'INFO'}, "Not enough meshes to compare.")
            return {'FINISHED'}

        # 2) GROUP BY VERTEX COUNT
        groups = {}
        for idx, e in enumerate(entries):
            groups.setdefault(e['vcount'], []).append(idx)

        # 3) PAIRWISE SYMMETRICAL KD‐CHECK FOR DUPLICATES
        adj = {i: set() for i in range(len(entries))}

        def is_duplicate(i, j):
            # Check that every point of entries[i] is near entries[j] and vice versa
            pts_i, kd_i = entries[i]['pts'], entries[i]['kd']
            pts_j, kd_j = entries[j]['pts'], entries[j]['kd']
            for co in pts_i:
                if kd_j.find(co)[2] > tol:
                    return False
            for co in pts_j:
                if kd_i.find(co)[2] > tol:
                    return False
            return True

        for vcount, idxs in groups.items():
            if len(idxs) < 2:
                continue
            for a in range(len(idxs) - 1):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if is_duplicate(i, j):
                        adj[i].add(j)
                        adj[j].add(i)

        # 4) BUILD CLUSTERS VIA GRAPH TRAVERSAL
        visited = set()
        clusters = []
        for start in range(len(entries)):
            if start in visited:
                continue
            stack = [start]
            comp = set()
            visited.add(start)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            clusters.append(comp)

        # 5) SELECT WHICH TO DELETE
        to_delete = []
        for comp in clusters:
            if len(comp) < 2:
                continue
            objs = [entries[i] for i in comp]
            normal_objs = [e for e in objs if e['has_normal']]
            matonly_objs = [e for e in objs if not e['has_normal']]

            if normal_objs:
                # Preserve all normal‐material meshes; delete the mat_-only ones
                for e in matonly_objs:
                    to_delete.append(e['obj'])
            else:
                # All are mat_-only → pick one keeper, delete the rest
                keeper = None
                for e in matonly_objs:
                    name = e['obj'].name
                    # “Pure Ninja” naming convention: mesh_XXX without dots
                    if name.startswith("mesh_") and "." not in name:
                        keeper = e['obj']
                        break
                if keeper is None:
                    keeper = matonly_objs[0]['obj']
                for e in matonly_objs:
                    if e['obj'] is not keeper:
                        to_delete.append(e['obj'])

        # 6) BATCH DELETE
        if to_delete:
            import bpy
            bpy.ops.object.select_all(action='DESELECT')
            first_active = None
            for ob in to_delete:
                if ob.name in scene.objects:
                    ob.select_set(True)
                    if first_active is None:
                        context.view_layer.objects.active = ob
                        first_active = ob
            if first_active:
                bpy.ops.object.delete()
            self.report({'INFO'}, f"Removed {len(to_delete)} duplicates.")
        else:
            self.report({'INFO'}, "No duplicates found to remove.")

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

def find_best_match_in_vcount_group(candidate_world_pts, relevant_golden_targets, tolerance=0.07):
    """
    (Optimized) Finds the best matching golden geometry from a pre-filtered
    list where all meshes are known to have the same vertex count.
    """
    best_match_hash = None
    min_post_align_rmsd = float('inf')

    for golden_hash, golden_world_pts in relevant_golden_targets.items():
        # Vertex count is already guaranteed to match here
        M = NFIX_AlignmentLogic.solve_alignment(candidate_world_pts, golden_world_pts)
        if M.to_3x3().determinant() == 0:
            continue

        aligned_pts = (np.hstack([candidate_world_pts, np.ones((candidate_world_pts.shape[0],1))]) @ np.array(M).T)[:,:3]
        rmsd = float(np.sqrt(np.mean(np.sum((aligned_pts - golden_world_pts)**2, axis=1))))
        
        if rmsd < min_post_align_rmsd:
            min_post_align_rmsd = rmsd
            best_match_hash = golden_hash
            # Optimization: if we find a near-perfect match, stop searching this group
            if min_post_align_rmsd < 1e-5:
                break

    if best_match_hash and min_post_align_rmsd < tolerance:
        return best_match_hash, min_post_align_rmsd
    
    return None, min_post_align_rmsd

class NFIX_OT_CalculateAndBatchFix(Operator):
    bl_idname = "nfix.calculate_and_batch_fix"
    bl_label  = "Generate & Apply Fix"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        st = context.scene.ninjafix_settings
        return NUMPY_INSTALLED and st.source_obj and st.target_obj

    def execute(self, context):
        global INITIAL_ALIGN_MATRIX, PREFERRED_ANCHOR_GEOM_HASHES, SEEN_GEOM_HASHES
        global GOLDEN_TARGETS_BY_VCOUNT, GOLDEN_GEOMETRY_DB_BY_VCOUNT

        st = context.scene.ninjafix_settings
        depsgraph = context.evaluated_depsgraph_get()
        src_pts = NFIX_AlignmentLogic.get_evaluated_world_points(st.source_obj, depsgraph)
        tgt_pts = NFIX_AlignmentLogic.get_evaluated_world_points(st.target_obj, depsgraph)

        if src_pts.shape[0] == 0 or src_pts.shape[0] != tgt_pts.shape[0]:
            self.report({'ERROR'}, "Meshes must be valid and have identical vertex counts.")
            return {'CANCELLED'}

        M0 = NFIX_AlignmentLogic.solve_alignment(src_pts, tgt_pts)
        INITIAL_ALIGN_MATRIX = M0.copy()

        blend_file = bpy.data.filepath
        if blend_file:
            db = load_db(); db[blend_file] = {"matrix": mat_to_list(M0)}; save_db(db)

        PREFERRED_ANCHOR_GEOM_HASHES.add(get_geometry_hash(st.source_obj))

        first_set = current_ninja_set(context.scene)
        for name in first_set:
            ob = context.scene.objects.get(name)
            if is_ninja(ob):
                if "ninjafix_prev_matrix" not in ob: ob["ninjafix_prev_matrix"] = mat_to_list(ob.matrix_world)
                ob.matrix_world = M0 @ ob.matrix_world

        # Clear and build the new optimized databases
        GOLDEN_TARGETS_BY_VCOUNT.clear()
        GOLDEN_GEOMETRY_DB_BY_VCOUNT.clear()
        SEEN_GEOM_HASHES.clear()
        depsgraph.update()

        remix_objs = [o for o in context.scene.objects if is_remix(o)]
        cap1_objs  = [context.scene.objects.get(n) for n in first_set]
        for obj in remix_objs + cap1_objs:
            if not obj or obj.type != 'MESH' or not obj.data.vertices:
                continue
            geom_hash = get_geometry_hash(obj)
            if not geom_hash: continue

            v_count = len(obj.data.vertices)
            
            # Populate geometry DB, grouped by vertex count
            local_coords = np.empty(v_count * 3, dtype=np.float32)
            obj.data.vertices.foreach_get("co", local_coords)
            local_coords.shape = (v_count, 3)
            GOLDEN_GEOMETRY_DB_BY_VCOUNT.setdefault(v_count, {})[geom_hash] = local_coords

            # Populate targets DB, grouped by vertex count
            world_pts = NFIX_AlignmentLogic.get_evaluated_world_points(obj, depsgraph)
            if world_pts.shape[0] > 0:
                GOLDEN_TARGETS_BY_VCOUNT.setdefault(v_count, {})[geom_hash] = world_pts
            
            SEEN_GEOM_HASHES.add(geom_hash)

        print(f"Database initialized with {len(SEEN_GEOM_HASHES)} unique shapes.")
        if not any(c['name'] == "Capture 1" for c in CAPTURE_SETS):
            CAPTURE_SETS.insert(0, {'name': "Capture 1", 'objects': first_set, 'reference_mesh': st.source_obj.name})
            context.scene["ninjafix_capture_sets"] = json.dumps([{'name': c['name'], 'objects': list(c['objects']), 'reference_mesh': c.get('reference_mesh','')} for c in CAPTURE_SETS])

        self.report({'INFO'}, "Initial alignment applied, and DB auto-saved.")
        return {'FINISHED'}

class NFIX_OT_ForceCapture1(Operator):
    bl_idname = "nfix.force_capture1"
    bl_label = "Force Capture 1"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Correctly declare the global variables that are being modified.
        global CAPTURE_SETS, GOLDEN_TARGETS_BY_VCOUNT, GOLDEN_GEOMETRY_DB_BY_VCOUNT, SEEN_GEOM_HASHES
        scene = context.scene

        # Record all current Ninja meshes as Capture 1
        first_set = current_ninja_set(scene)
        CAPTURE_SETS.insert(0, {
            'name': "Capture 1",
            'objects': first_set,
            'reference_mesh': ''  # No reference mesh since this is the starting point
        })
        # Update the scene property for saving/reloading
        context.scene["ninjafix_capture_sets"] = json.dumps([
            {'name': c['name'], 'objects': list(c['objects']), 'reference_mesh': c.get('reference_mesh','')}
            for c in CAPTURE_SETS
        ])

        # --- CORRECTED DATABASE INITIALIZATION ---
        # Build in-memory blueprint (golden targets) from the entire current scene state.
        
        # 1. Clear the correct, new data structures.
        GOLDEN_TARGETS_BY_VCOUNT.clear()
        GOLDEN_GEOMETRY_DB_BY_VCOUNT.clear()
        SEEN_GEOM_HASHES.clear()
        
        depsgraph = context.evaluated_depsgraph_get()
        print("\nForcing Capture 1: Building new Golden Database from all visible meshes...")

        # 2. Populate the databases using the new _BY_VCOUNT structure.
        for obj in scene.objects:
            if obj.type != 'MESH' or not obj.data or not obj.data.vertices:
                continue
            
            geom_hash = get_geometry_hash(obj)
            if not geom_hash:
                continue

            v_count = len(obj.data.vertices)

            # Populate geometry DB with local-space coordinates
            local_coords = np.empty(v_count * 3, dtype=np.float32)
            obj.data.vertices.foreach_get("co", local_coords)
            local_coords.shape = (v_count, 3)
            GOLDEN_GEOMETRY_DB_BY_VCOUNT.setdefault(v_count, {})[geom_hash] = local_coords
            
            # Populate targets DB with world-space coordinates
            t_pts = NFIX_AlignmentLogic.get_evaluated_world_points(obj, depsgraph)
            if t_pts.shape[0] > 0:
                GOLDEN_TARGETS_BY_VCOUNT.setdefault(v_count, {})[geom_hash] = t_pts
            
            SEEN_GEOM_HASHES.add(geom_hash)

        print(f"Database created with {len(SEEN_GEOM_HASHES)} unique shapes.")
        self.report({'INFO'}, "Forced Capture 1 and built new blueprint from scene.")
        return {'FINISHED'}

# =================================================================================================
# STAGE 4: PROCESS ALL OTHER CAPTURES
# =================================================================================================
class NFIX_OT_ProcessAllCaptures(Operator):
    bl_idname = "nfix.process_all_captures"
    bl_label = "Process All Captures"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return NUMPY_INSTALLED and len(CAPTURE_SETS) > 1 and bool(GOLDEN_TARGETS_BY_VCOUNT)

    def execute(self, context):
        global INITIAL_ALIGN_MATRIX, PREFERRED_ANCHOR_GEOM_HASHES, SEEN_GEOM_HASHES
        global GOLDEN_TARGETS_BY_VCOUNT, GOLDEN_GEOMETRY_DB_BY_VCOUNT

        import numpy as np
        from mathutils import Matrix, kdtree
        from collections import Counter, defaultdict

        if not GOLDEN_TARGETS_BY_VCOUNT:
            self.report({'ERROR'}, "No cached target data found. Run Stage 2 first.")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        matrix_tol = 0.01
        alignment_quality_tolerance = 0.07
        spatial_coincidence_tolerance = 0.005

        # --- PHASE 0: PRE-ANALYSIS OF DUPLICATE GEOMETRIES ---
        # This part remains the same.
        print("\n--- Phase 0: Analyzing scene for safe duplicate geometries ---")
        all_objects_by_hash = defaultdict(list)
        for cap in CAPTURE_SETS:
            for obj_name in cap['objects']:
                obj = context.scene.objects.get(obj_name)
                if obj:
                    geom_hash = get_geometry_hash(obj)
                    if geom_hash: all_objects_by_hash[geom_hash].append(obj)
        
        safe_duplicate_hashes = set()
        world_pts_cache = {}
        def get_cached_world_points(obj):
            if obj.name not in world_pts_cache:
                world_pts_cache[obj.name] = NFIX_AlignmentLogic.get_evaluated_world_points(obj, depsgraph)
            return world_pts_cache[obj.name]

        def are_coincident(pts_ref, pts_other, tolerance):
            if pts_ref.shape != pts_other.shape or pts_ref.shape[0] == 0: return False
            kd = kdtree.KDTree(len(pts_ref))
            for i, v in enumerate(pts_ref): kd.insert(v, i)
            kd.balance()
            max_dist_sq = tolerance ** 2
            for point in pts_other:
                _, _, dist_sq = kd.find(point)
                if dist_sq > max_dist_sq: return False
            return True

        for geom_hash, objects in all_objects_by_hash.items():
            if len(objects) > 1:
                is_safe = True
                ref_pts = get_cached_world_points(objects[0])
                if ref_pts.shape[0] == 0: continue
                for i in range(1, len(objects)):
                    other_pts = get_cached_world_points(objects[i])
                    if not are_coincident(ref_pts, other_pts, spatial_coincidence_tolerance):
                        is_safe = False
                        break
                if is_safe: safe_duplicate_hashes.add(geom_hash)
        
        # --- PHASE 1: Find transforms for each capture without applying them or changing state ---
        transforms_to_apply = {}
        unprocessed_captures = [cap for cap in CAPTURE_SETS[1:] if not cap.get('reference_mesh')]
        
        for cap in unprocessed_captures:
            print(f"\n{'='*25} FINDING ALIGNMENT FOR CAPTURE: {cap['name']} {'='*25}")
            
            all_ninja_objs = [context.scene.objects.get(nm) for nm in cap['objects'] if is_ninja(context.scene.objects.get(nm))]
            geom_cache = {o.name: get_geometry_hash(o) for o in all_ninja_objs if o}
            pts_cache = {o.name: get_cached_world_points(o) for o in all_ninja_objs if o}
            hashes_in_this_capture = Counter(h for h in geom_cache.values() if h)
            
            valid_candidates = []
            for name, pts in pts_cache.items():
                if pts is None or pts.shape[0] < 4: continue
                g_hash = geom_cache.get(name, '')
                if not g_hash or hashes_in_this_capture[g_hash] != 1: continue
                if g_hash in safe_duplicate_hashes or len(all_objects_by_hash[g_hash]) == 1:
                    valid_candidates.append(name)

            print(f"Found {len(valid_candidates)} valid anchor candidates. Pre-calculating alignment data...")

            # --- NEW OPTIMIZATION: Pre-calculate alignment data for each candidate ONCE ---
            candidate_data_cache = {}
            for name in valid_candidates:
                pts = pts_cache[name]
                vcount = pts.shape[0]
                if vcount not in GOLDEN_TARGETS_BY_VCOUNT: continue
                
                match_hash, _ = find_best_match_in_vcount_group(pts, GOLDEN_TARGETS_BY_VCOUNT[vcount], alignment_quality_tolerance)
                if match_hash:
                    target_pts = GOLDEN_TARGETS_BY_VCOUNT[vcount][match_hash]
                    matrix = NFIX_AlignmentLogic.solve_alignment(pts, target_pts)
                    candidate_data_cache[name] = {
                        'matrix': matrix,
                        'golden_hash': match_hash,
                        'centroid': np.mean(pts, axis=0)
                    }

            print(f"Cached data for {len(candidate_data_cache)} candidates. Searching for best pair...")
            
            # --- Fast Pairing: Use the cache to find the best-separated pair ---
            best_pair_info = None
            max_separation_dist = -1.0
            
            cached_candidates = list(candidate_data_cache.keys())

            for i, primary_name in enumerate(cached_candidates):
                primary_data = candidate_data_cache[primary_name]
                for j in range(i + 1, len(cached_candidates)):
                    secondary_name = cached_candidates[j]
                    secondary_data = candidate_data_cache[secondary_name]

                    # Check 1: Anchors must match different golden meshes
                    if primary_data['golden_hash'] == secondary_data['golden_hash']:
                        continue
                    
                    # Check 2: Check for transform consensus using cached matrices
                    if np.linalg.norm(np.array(primary_data['matrix']) - np.array(secondary_data['matrix'])) < matrix_tol:
                        # This is a valid pair. Check if it's the best one.
                        separation_dist = np.linalg.norm(primary_data['centroid'] - secondary_data['centroid'])
                        
                        if separation_dist > max_separation_dist:
                            max_separation_dist = separation_dist
                            best_pair_info = {
                                'matrix': primary_data['matrix'], 
                                'ref_mesh': primary_name, 
                                'val_mesh': secondary_name,
                                'dist': separation_dist
                            }
            
            if best_pair_info:
                print(f"SUCCESS: Selected best anchor pair ('{best_pair_info['ref_mesh']}', '{best_pair_info['val_mesh']}') with separation distance of {best_pair_info['dist']:.4f}.")
                transforms_to_apply[cap['name']] = {
                    'matrix': best_pair_info['matrix'], 'ref_mesh': best_pair_info['ref_mesh'], 'val_mesh': best_pair_info['val_mesh'],
                    'objects': cap['objects'], 'geom_cache': geom_cache
                }
            else:
                print(f"FAILURE: Could not find any valid, verified anchor pair for '{cap['name']}'.")
        
        # --- PHASE 2 & 3: Apply transforms and update database (No changes here) ---
        aligned_count = len(transforms_to_apply)
        if aligned_count > 0:
            print(f"\n--- Phase 2: Applying {aligned_count} found transforms ---")
            for cap_name, data in transforms_to_apply.items():
                matrix, ref_mesh, val_mesh = data['matrix'], data['ref_mesh'], data['val_mesh']
                cap_to_update = next((c for c in CAPTURE_SETS if c['name'] == cap_name), None)
                if cap_to_update:
                    cap_to_update['reference_mesh'], cap_to_update['validation_mesh'] = ref_mesh, val_mesh
                for nm in data['objects']:
                    ob = context.scene.objects.get(nm)
                    if is_ninja(ob):
                        if "ninjafix_prev_matrix" not in ob: ob["ninjafix_prev_matrix"] = mat_to_list(ob.matrix_world)
                        ob.matrix_world = matrix @ ob.matrix_world

            print("\n--- Phase 3: Updating Golden Database with new geometries ---")
            depsgraph.update()
            newly_added_hashes = 0
            for cap_name, data in transforms_to_apply.items():
                for nm in data['objects']:
                    ob = context.scene.objects.get(nm)
                    if not (is_ninja(ob) and ob.data and ob.data.vertices): continue
                    geom_hash = data['geom_cache'].get(ob.name)
                    if not geom_hash or geom_hash in SEEN_GEOM_HASHES: continue
                    v_count, local_coords = len(ob.data.vertices), np.empty(len(ob.data.vertices)*3,'f')
                    ob.data.vertices.foreach_get("co",local_coords); local_coords.shape=(-1,3)
                    GOLDEN_GEOMETRY_DB_BY_VCOUNT.setdefault(v_count, {})[geom_hash] = local_coords
                    newly_aligned_pts = NFIX_AlignmentLogic.get_evaluated_world_points(ob, depsgraph)
                    if newly_aligned_pts.shape[0] > 0:
                        GOLDEN_TARGETS_BY_VCOUNT.setdefault(v_count, {})[geom_hash] = newly_aligned_pts
                        SEEN_GEOM_HASHES.add(geom_hash)
                        newly_added_hashes += 1
            print(f"Added {newly_added_hashes} new unique geometries to the database.")

        self.report({'INFO'}, f"Processing complete: {aligned_count} aligned, {len(unprocessed_captures) - aligned_count} skipped.")
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

        icon_items = bpy.types.UILayout.bl_rna.functions['operator'].parameters['icon'].enum_items.keys()
        rotate_icon = 'ROTATE' if 'ROTATE' in icon_items else 'FILE_TICK'

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

        box = layout.box()
        box.label(text="Stage 2: Initial Alignment", icon='PLAY')
        box.enabled = bool(st.source_obj and st.target_obj)
        box.operator('nfix.calculate_and_batch_fix', text="Generate & Apply Fix", icon=rotate_icon)

        box = layout.box()
        box.label(text="Stage 3: Manage Captures", icon='BOOKMARKS')
        box.prop(st, "auto_capture")
        box.prop(st, "import_folders", text="Import From Folders", icon='FILE_FOLDER')
        if st.import_folders:
            box.prop(st, "flip_geometry", text="Flip Geometry")
            box.prop(context.scene, "nfix_parent_dir", text="Parent Folder")
            box.operator("nfix.scan_capture_folders", icon='FILE_FOLDER')
            box.template_list("NFIX_UL_FolderList", "", context.scene, "nfix_folder_items", context.scene, "nfix_folder_index", rows=4)
            box.operator("nfix.import_selected_folders", icon='IMPORT')

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

        box = layout.box()
        box.label(text="Stage 4: Process All & Cleanup", icon='FILE_TICK')
        box.operator('nfix.process_all_captures', text="Process All Captures", icon='CHECKMARK')
        box.operator('nfix.undo_last_transform', text="Undo Last Transform", icon='LOOP_BACK')
        box.operator('nfix.remove_mat_duplicates', text="Remove mat_ Duplicates", icon='TRASH')

        # --- DEBUGGING TOOLS (MODIFIED) ---
        box = layout.box()
        # The column(align=True) has been removed from here.
        # The operators are now drawn directly onto the box for standard spacing.
        box.label(text="Debugging Tools", icon='GHOST_ENABLED')
        box.operator("nfix.debug_shape_compare", text="Generate Anchor Report")
        box.operator("nfix.select_by_texture")

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
    NFIX_OT_DebugShapeCompare,
    NFIX_OT_SelectByTextureHash, # ADD THIS CLASS NAME
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
