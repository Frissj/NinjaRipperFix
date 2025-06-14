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
GOLDEN_TARGETS = {} # NEW: To store the 'snapshot' of target points
PREFERRED_ANCHOR_HASHES = set()
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

def images_are_close(arr1, arr2, tol=1e-4):
    """
    Return True if two flattened pixel buffers differ by less than
    tol * number_of_elements. Supports both numpy arrays and Python array.array.
    """
    import numpy as np

    # Convert to numpy arrays if necessary
    a1 = np.asarray(arr1, dtype=np.float32)
    a2 = np.asarray(arr2, dtype=np.float32)

    # Must have the same shape/length
    if a1.shape != a2.shape:
        return False

    # Euclidean norm of the difference
    diff = np.linalg.norm(a1 - a2)

    # Compare to tol * total number of floats
    return diff < tol * a1.size


def get_individual_texture_hashes(obj):
    """
    Finds all ImageTexture nodes driving Base Color on used slots,
    reads their full pixel buffers once, and returns their MD5 hashes.
    """
    hashes = set()
    try:
        used_idxs = {p.material_index for p in obj.data.polygons}
    except Exception:
        used_idxs = set()

    for idx, slot in enumerate(obj.material_slots):
        if idx not in used_idxs or not slot.material or not slot.material.use_nodes:
            continue

        output_node = next(
            (n for n in slot.material.node_tree.nodes
             if n.type == 'OUTPUT_MATERIAL' and n.is_active_output),
            None
        )
        if not output_node or not output_node.inputs['Surface'].is_linked:
            continue

        bsdf = output_node.inputs['Surface'].links[0].from_node
        if bsdf.type != 'BSDF_PRINCIPLED':
            continue

        base_in = bsdf.inputs.get('Base Color')
        if not base_in or not base_in.is_linked:
            continue

        def find_tex_node(node, visited):
            if node in visited:
                return None
            visited.add(node)
            if node.type == 'TEX_IMAGE':
                return node
            for inp in node.inputs:
                if inp.is_linked:
                    found = find_tex_node(inp.links[0].from_node, visited)
                    if found:
                        return found
            return None

        tex_node = find_tex_node(base_in.links[0].from_node, set())
        if not tex_node or not getattr(tex_node, 'image', None):
            continue

        img = tex_node.image
        hashes.add(hash_image_pixels(img))

    return hashes

def hash_image_pixels(img):
    """
    Read img.pixels in one go via foreach_get into a Python array,
    MD5 the raw bytes, and cache both the hash and the raw buffer
    for later fuzzy matching.
    """
    import array
    global TEXTURE_PIXEL_CACHE

    name = img.name
    if name in _TEXTURE_HASH_CACHE:
        return _TEXTURE_HASH_CACHE[name]

    # compute buffer size (RGBA)
    w, h = img.size
    total = w * h * 4

    # bulk-copy into a Python float array
    buf = array.array('f', [0.0]) * total
    img.pixels.foreach_get(buf)

    # hash the raw bytes
    hsh = hashlib.md5(buf.tobytes()).hexdigest()

    # cache both hash and buffer for fuzzy matching
    _TEXTURE_HASH_CACHE[name] = hsh
    TEXTURE_PIXEL_CACHE[hsh] = buf

    return hsh

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

    @staticmethod
    def get_texture_hash(obj):
        """
        Grab the first Base-Color image on any used material slot,
        hash its pixels (using our cached, bulk-foreach_get helper),
        and return that MD5. Fast and cached.
        """
        try:
            used_idxs = {p.material_index for p in obj.data.polygons}
        except Exception:
            used_idxs = set()

        for idx, slot in enumerate(obj.material_slots):
            if idx not in used_idxs or not slot.material or not slot.material.use_nodes:
                continue

            output_node = next(
                (n for n in slot.material.node_tree.nodes
                 if n.type == 'OUTPUT_MATERIAL' and n.is_active_output),
                None
            )
            if not output_node or not output_node.inputs['Surface'].is_linked:
                continue

            bsdf = output_node.inputs['Surface'].links[0].from_node
            if bsdf.type != 'BSDF_PRINCIPLED':
                continue

            base_in = bsdf.inputs.get('Base Color')
            if not base_in or not base_in.is_linked:
                continue

            def find_tex_node(node, visited):
                if node in visited:
                    return None
                visited.add(node)
                if node.type == 'TEX_IMAGE':
                    return node
                for inp in node.inputs:
                    if inp.is_linked:
                        found = find_tex_node(inp.links[0].from_node, visited)
                        if found:
                            return found
                return None

            tex_node = find_tex_node(base_in.links[0].from_node, set())
            if not tex_node or not getattr(tex_node, 'image', None):
                continue

            # Use our fast, cached pixel-hash helper:
            return hash_image_pixels(tex_node.image)

        return ""

class NFIX_OT_DebugShapeCompare(Operator):
    """
    The primary diagnostic tool.
    - 1 mesh: Full Anchor Candidate Report.
    - 2 meshes: Precise shape difference (RMSD) analysis.
    - 4 meshes (2 Ninja, 2 Remix): Full anchor pair validation, including fuzzy-texture matching.
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
            obj = selected[0]
            report_lines = [
                "\n" + "="*50,
                " NINJAFIX ANCHOR CANDIDATE REPORT",
                "="*50,
                f"Analyzing Candidate: {obj.name}\n"
            ]

            report_lines.append("--- 1. Uniqueness ---")
            if not NON_UNIQUE_GEOM_HASHES:
                print("ERROR: Run Stage 2 to build uniqueness database first.")
                return {'CANCELLED'}
            gh = get_geometry_hash(obj)
            unique = gh not in NON_UNIQUE_GEOM_HASHES
            report_lines.append(f"Result: {'UNIQUE' if unique else 'NON-UNIQUE'}")
            report_lines.append(f"Verdict: {'PASSED.' if unique else 'INVALID ANCHOR.'}")
            report_lines.append("")

            report_lines.append("--- 2. Golden Counterpart ---")
            th = NFIX_AlignmentLogic.get_texture_hash(obj)
            has_gold = th in GOLDEN_TARGETS
            report_lines.append(f"Result: {'Found' if has_gold else 'NOT Found'}")
            report_lines.append(f"Verdict: {'PASSED.' if has_gold else 'INVALID ANCHOR.'}")
            report_lines.append("")

            final = "POOR"
            if unique and has_gold:
                report_lines.append("--- 3. Shape Quality ---")
                pts = NFIX_AlignmentLogic.get_evaluated_world_points(obj, depsgraph)
                best = float('inf')
                for tgt in GOLDEN_TARGETS[th]:
                    if pts.shape == tgt.shape:
                        best = min(best, calculate_shape_difference(pts, tgt))
                report_lines.append(f"Best RMSD: {best:.8f}")
                if best < 0.01:
                    report_lines.append("Verdict: PASSED. Shape difference below tolerance.")
                    final = "GOOD"
                else:
                    report_lines.append("Verdict: FAILED. Shape difference too high.")
            summary = f"This is a {final} anchor candidate."
            report_lines += ["\n" + "="*50, f"  FINAL VERDICT: {summary}", "="*50]
            for line in report_lines:
                print(line)
            self.report({'INFO'}, f"Report Generated. Final Verdict: {summary}")
            return {'FINISHED'}

        # --- 2 objects: Precise Shape Comparison ---
        elif len(selected) == 2:
            a, b = selected
            if is_ninja(a) and is_remix(b):
                ninja_obj, remix_obj = a, b
            elif is_ninja(b) and is_remix(a):
                ninja_obj, remix_obj = b, a
            else:
                print("ERROR: Must select one Ninja and one Remix mesh.")
                return {'CANCELLED'}

            pts_n = NFIX_AlignmentLogic.get_evaluated_world_points(ninja_obj, depsgraph)
            pts_r = NFIX_AlignmentLogic.get_evaluated_world_points(remix_obj, depsgraph)
            if pts_n.shape[0] < 3 or pts_r.shape[0] < 3 or pts_n.shape[0] != pts_r.shape[0]:
                print("ERROR: Need ≥3 verts and equal counts.")
                return {'CANCELLED'}

            M = NFIX_AlignmentLogic.solve_alignment(pts_n, pts_r)
            ones = np.ones((pts_n.shape[0],1))
            aligned = (np.hstack([pts_n.astype(np.float64), ones]) @ np.array(M).T)[:,:3]
            rmsd = float(np.sqrt(np.mean(np.sum((aligned - pts_r)**2, axis=1))))
            print("\n" + "="*50)
            print(" NINJAFIX FINAL SHAPE ANALYSIS")
            print("="*50)
            print(f"Ninja mesh: {ninja_obj.name}")
            print(f"Remix mesh: {remix_obj.name}\n")
            print(f"Affine-Aligned RMSD: {rmsd:.8f}\n")
            conclusion = "IDENTICAL" if rmsd < 1e-6 else f"DIFFERENT (RMSD: {rmsd:.6f})"
            print(f"Conclusion: {conclusion}")
            print("="*50)
            self.report({'INFO'}, f"Analysis Complete. Summary: {conclusion}")
            return {'FINISHED'}

        # --- 4 objects: Texture-Pairing + Fuzzy Match + Shape Validate ---
        elif len(selected) == 4:
            ninjas = [o for o in selected if is_ninja(o)]
            remixes = [o for o in selected if is_remix(o)]
            if len(ninjas)!=2 or len(remixes)!=2:
                print("ERROR: Must select exactly 2 Ninja and 2 Remix meshes.")
                return {'CANCELLED'}

            print("\n" + "="*60)
            print(" NINJAFIX INDIVIDUAL TEXTURE HASH ANALYSIS")
            print("="*60)
            ninja_hashes = [get_individual_texture_hashes(n) for n in ninjas]
            remix_hashes = [get_individual_texture_hashes(r) for r in remixes]
            for i,n in enumerate(ninjas):
                print(f"\nObject: '{n.name}' (Ninja) → {ninja_hashes[i]}")
            for i,r in enumerate(remixes):
                print(f"\nObject: '{r.name}' (Remix) → {remix_hashes[i]}")
            print("\n" + "="*60)

            pairs = []
            used_n = set()
            used_r = set()
            for ri, rhset in enumerate(remix_hashes):
                for ni, nhset in enumerate(ninja_hashes):
                    if ni in used_n: continue
                    common = rhset & nhset
                    # fuzzy fallback
                    if not common:
                        for rh in rhset:
                            for nh in nhset:
                                arr_r = TEXTURE_PIXEL_CACHE.get(rh)
                                arr_n = TEXTURE_PIXEL_CACHE.get(nh)
                                if arr_r is not None and arr_n is not None and images_are_close(arr_r, arr_n):
                                    common = {rh}
                                    print(f"[FUZZY] Treating remix-hash {rh} ≈ ninja-hash {nh}")
                                    break
                            if common:
                                break
                    if common:
                        pairs.append((ninjas[ni], remixes[ri]))
                        used_n.add(ni)
                        used_r.add(ri)
                        break

            if len(pairs)!=2:
                print("ERROR: Could not form two texture-pairs.")
                return {'CANCELLED'}

            print("SUCCESS: Formed pairs:", [(n.name, r.name) for n,r in pairs])

            # shape-check each
            results = []
            tol = 0.07
            for idx,(n,r) in enumerate(pairs):
                pts1 = NFIX_AlignmentLogic.get_evaluated_world_points(n, depsgraph)
                pts2 = NFIX_AlignmentLogic.get_evaluated_world_points(r, depsgraph)
                if pts1.shape!=pts2.shape or pts1.shape[0]<4:
                    results.append((idx, None, "Vertex Mismatch"))
                    continue
                M = NFIX_AlignmentLogic.solve_alignment(pts1, pts2)
                aligned = (np.hstack([pts1, np.ones((pts1.shape[0],1))]) @ np.array(M).T)[:,:3]
                rmsd = float(np.sqrt(np.mean(np.sum((aligned-pts2)**2, axis=1))))
                verdict = "PASSED" if rmsd<tol else f"FAILED (RMSD {rmsd:.6f})"
                results.append((idx, rmsd, verdict))

            # consensus
            final_ok = all(r[2]=="PASSED" for r in results)
            print("\n" + "="*60)
            print(" NINJAFIX ANCHOR PAIR VALIDATION REPORT")
            print("="*60)
            for i,(rmsd,verdict) in enumerate([(r[1],r[2]) for r in results]):
                n,r = pairs[i]
                print(f"Pair {i+1}: '{n.name}'→'{r.name}' | RMSD: {rmsd:.8f} → {verdict}")
            if final_ok:
                print("FINAL VERDICT: VALID ANCHOR PAIR")
            else:
                print("FINAL VERDICT: INVALID ANCHOR PAIR")
            print("="*60)

            self.report({'INFO'}, "Validation Complete. See System Console for full report.")
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
        global INITIAL_ALIGN_MATRIX, GOLDEN_TARGETS, PREFERRED_ANCHOR_HASHES, SEEN_GEOM_HASHES

        st = context.scene.ninjafix_settings
        depsgraph = context.evaluated_depsgraph_get()

        # 1) Compute our alignment matrix
        src_pts = NFIX_AlignmentLogic.get_evaluated_world_points(st.source_obj, depsgraph)
        tgt_pts = NFIX_AlignmentLogic.get_evaluated_world_points(st.target_obj, depsgraph)

        if src_pts.shape[0] == 0 or src_pts.shape[0] != tgt_pts.shape[0]:
            self.report({'ERROR'}, "Meshes must be valid and have identical vertex counts.")
            return {'CANCELLED'}

        M0 = NFIX_AlignmentLogic.solve_alignment(src_pts, tgt_pts)
        INITIAL_ALIGN_MATRIX = M0.copy()

        # 2) Persist it immediately to disk
        blend_file = bpy.data.filepath
        if blend_file:
            db = load_db()
            loc, rot, scale = M0.decompose()
            db[blend_file] = {
                "scale":  [scale.x, scale.y, scale.z],
                "matrix": mat_to_list(M0)
            }
            save_db(db)  # ← Auto-save your stage-2 result on disk

        # 3) Mark our preferred anchor
        PREFERRED_ANCHOR_HASHES.add(
            NFIX_AlignmentLogic.get_texture_hash(st.source_obj)
        )

        # 4) Apply the transform in-place to all current Ninja meshes
        first_set = current_ninja_set(context.scene)
        for name in first_set:
            ob = context.scene.objects.get(name)
            if is_ninja(ob):
                if "ninjafix_prev_matrix" not in ob:
                    ob["ninjafix_prev_matrix"] = mat_to_list(ob.matrix_world)
                ob.matrix_world = M0 @ ob.matrix_world

        # 5) Rebuild dynamic “golden” database from both the Remix and this first capture
        GOLDEN_TARGETS.clear()
        SEEN_GEOM_HASHES.clear()
        depsgraph.update()

        # collect all targets
        remix_objs = [o for o in context.scene.objects if is_remix(o)]
        cap1_objs  = [context.scene.objects.get(n) for n in first_set]
        for obj in remix_objs + cap1_objs:
            if not obj or obj.type != 'MESH':
                continue
            tex_hash = NFIX_AlignmentLogic.get_texture_hash(obj)
            pts = NFIX_AlignmentLogic.get_evaluated_world_points(obj, depsgraph)
            if pts.shape[0] > 0:
                GOLDEN_TARGETS.setdefault(tex_hash, []).append(pts)

            geom_hash = get_geometry_hash(obj)
            if geom_hash:
                SEEN_GEOM_HASHES.add(geom_hash)

        print(f"Database initialized with {len(SEEN_GEOM_HASHES)} unique shapes.")

        # 6) Register the first capture if it wasn’t already
        if not any(c['name'] == "Capture 1" for c in CAPTURE_SETS):
            CAPTURE_SETS.insert(0, {
                'name': "Capture 1",
                'objects': first_set,
                'reference_mesh': st.source_obj.name
            })
            context.scene["ninjafix_capture_sets"] = json.dumps([
                {
                    'name': c['name'],
                    'objects': list(c['objects']),
                    'reference_mesh': c.get('reference_mesh','')
                }
                for c in CAPTURE_SETS
            ])

        self.report({'INFO'}, "Initial alignment applied, and DB auto-saved.")
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
    bl_idname = "nfix.process_all_captures"
    bl_label = "Process All Captures"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return NUMPY_INSTALLED and len(CAPTURE_SETS) > 1 and bool(GOLDEN_TARGETS)

    def execute(self, context):
        global INITIAL_ALIGN_MATRIX, GOLDEN_TARGETS, PREFERRED_ANCHOR_HASHES, SEEN_GEOM_HASHES, TEXTURE_PIXEL_CACHE

        import numpy as np
        from mathutils import Matrix
        from collections import Counter, defaultdict

        if not GOLDEN_TARGETS:
            self.report({'ERROR'}, "No cached target data found. Run Stage 2 first.")
            return {'CANCELLED'}

        # 1) Pre-cache every golden texture's pixel buffer
        for gh in list(GOLDEN_TARGETS.keys()):
            if gh not in TEXTURE_PIXEL_CACHE:
                for ob in context.scene.objects:
                    if ob.type == 'MESH':
                        hashes = get_individual_texture_hashes(ob)
                        if gh in hashes:
                            break

        # 2) Build a list of (golden_hash, pixel_array, target_point_lists) for fast fuzzy
        golden_entries = [
            (gh, TEXTURE_PIXEL_CACHE[gh], GOLDEN_TARGETS[gh])
            for gh in GOLDEN_TARGETS
            if gh in TEXTURE_PIXEL_CACHE
        ]

        depsgraph = context.evaluated_depsgraph_get()
        matrix_tol = 0.01
        shape_match_tol = 0.07
        aligned_count = 0
        skipped_count = 0
        successful_anchors = []

        # 3) Compute 10% of scene diagonal once
        scene_pts = []
        for capx in CAPTURE_SETS[1:]:
            for nm in capx['objects']:
                ob = context.scene.objects.get(nm)
                if ob and ob.type == 'MESH':
                    pts = NFIX_AlignmentLogic.get_evaluated_world_points(ob, depsgraph)
                    if pts.size:
                        scene_pts.append(pts)
        if scene_pts:
            all_pts = np.vstack(scene_pts)
            bb_min, bb_max = all_pts.min(axis=0), all_pts.max(axis=0)
            min_anchor_dist = np.linalg.norm(bb_max - bb_min) * 0.10
        else:
            min_anchor_dist = 0.0

        # 4) Process each capture
        for cap in CAPTURE_SETS[1:]:
            if cap.get('reference_mesh'):
                print(f"Skipping capture '{cap['name']}' (already processed)")
                continue

            cap['reference_mesh'] = ''
            cap['validation_mesh'] = ''
            print(f"\n{'='*25} PROCESSING CAPTURE: {cap['name']} {'='*25}")

            failure_reasons = defaultdict(set)
            consensus_failures = []

            # 4.a) Gather all Ninja objects and caches
            all_ninja_objs = [
                context.scene.objects.get(nm)
                for nm in cap['objects']
                if is_ninja(context.scene.objects.get(nm))
            ]
            pts_cache = {
                o.name: NFIX_AlignmentLogic.get_evaluated_world_points(o, depsgraph)
                for o in all_ninja_objs if o
            }
            geom_cache = {
                o.name: get_geometry_hash(o)
                for o in all_ninja_objs if o
            }
            tex_hash_cache = {
                o.name: NFIX_AlignmentLogic.get_texture_hash(o)
                for o in all_ninja_objs if o
            }
            pixel_cache = {
                name: TEXTURE_PIXEL_CACHE.get(tex_hash_cache[name])
                for name in tex_hash_cache
            }

            # 4.b) Build geometry histogram across other captures
            hist = Counter()
            for other in (c for c in CAPTURE_SETS if c is not cap):
                for nm in other['objects']:
                    ob = context.scene.objects.get(nm)
                    if ob and ob.type == 'MESH':
                        h = get_geometry_hash(ob)
                        if h:
                            hist[h] += 1

            # 4.c) Determine candidate order
            preferred = [
                name for name in tex_hash_cache
                if tex_hash_cache[name] in PREFERRED_ANCHOR_HASHES
            ]
            others = [
                name for name in tex_hash_cache
                if name not in preferred
            ]
            others.sort(
                key=lambda n: pts_cache.get(n, np.array([])).shape[0],
                reverse=True
            )
            candidates = preferred + others

            final_matrix = None

            # 4.d) Try each primary candidate
            for primary_name in candidates:
                if final_matrix:
                    break

                # Check uniqueness and vertex count
                geom_h1 = geom_cache.get(primary_name)
                if not geom_h1 or hist[geom_h1] > 1:
                    failure_reasons["Primary: Not Unique"].add(primary_name)
                    continue
                pts1 = pts_cache.get(primary_name)
                if pts1 is None or pts1.shape[0] < 4:
                    failure_reasons["Primary: Too Few Verts"].add(primary_name)
                    continue

                # Fuzzy-aware golden lookup for primary
                phash = tex_hash_cache[primary_name]
                arr_p = pixel_cache.get(primary_name)
                match_primary = None
                if phash in GOLDEN_TARGETS:
                    match_primary = phash
                elif arr_p is not None:
                    for gh, arr_g, tgt_list in golden_entries:
                        if images_are_close(arr_p, arr_g):
                            match_primary = gh
                            print(f"[FUZZY] Treating primary-hash {phash} ≈ golden-hash {gh} for '{primary_name}'")
                            break
                if not match_primary:
                    failure_reasons["Primary: No Golden Counterpart"].add(primary_name)
                    continue

                centroid1 = np.mean(pts1, axis=0)

                # 4.e) Build secondary list
                secondaries = []
                for secondary_name in candidates:
                    if secondary_name == primary_name:
                        continue
                    geom_h2 = geom_cache.get(secondary_name)
                    if not geom_h2 or hist[geom_h2] > 1:
                        continue
                    shash = tex_hash_cache[secondary_name]
                    if shash == phash:
                        continue
                    secondaries.append(secondary_name)
                secondaries.sort(
                    key=lambda n: np.linalg.norm(np.mean(pts_cache.get(n, np.array([])), axis=0) - centroid1),
                    reverse=True
                )

                # 4.f) Try each secondary
                for secondary_name in secondaries:
                    if final_matrix:
                        break

                    pts2 = pts_cache.get(secondary_name)
                    if pts2 is None or pts2.shape[0] < 4:
                        failure_reasons["Secondary: Too Few Verts"].add(secondary_name)
                        continue

                    # Fuzzy-aware golden lookup for secondary
                    shash = tex_hash_cache[secondary_name]
                    arr_s = pixel_cache.get(secondary_name)
                    match_secondary = None
                    if shash in GOLDEN_TARGETS:
                        match_secondary = shash
                    elif arr_s is not None:
                        for gh, arr_g, tgt_list in golden_entries:
                            if images_are_close(arr_s, arr_g):
                                match_secondary = gh
                                print(f"[FUZZY] Treating secondary-hash {shash} ≈ golden-hash {gh} for '{secondary_name}'")
                                break
                    if not match_secondary:
                        failure_reasons["Secondary: No Golden Counterpart"].add(secondary_name)
                        continue

                    centroid2 = np.mean(pts2, axis=0)
                    if np.linalg.norm(centroid2 - centroid1) < min_anchor_dist:
                        failure_reasons["Secondary: Too Close to Primary"].add(secondary_name)
                        continue

                    # 4.g) Compute alignments
                    M1 = None
                    for tgt_pts1 in GOLDEN_TARGETS[match_primary]:
                        if pts1.shape != tgt_pts1.shape:
                            continue
                        m1 = NFIX_AlignmentLogic.solve_alignment(pts1, tgt_pts1)
                        if m1.to_3x3().determinant() > 0:
                            homog1 = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
                            aligned1 = (homog1 @ np.array(m1).T)[:, :3]
                            if np.sqrt(np.mean(np.sum((aligned1 - tgt_pts1)**2, axis=1))) < shape_match_tol:
                                M1 = m1
                                break

                    if not M1:
                        continue

                    M2 = None
                    for tgt_pts2 in GOLDEN_TARGETS[match_secondary]:
                        if pts2.shape != tgt_pts2.shape:
                            continue
                        m2 = NFIX_AlignmentLogic.solve_alignment(pts2, tgt_pts2)
                        if m2.to_3x3().determinant() > 0:
                            homog2 = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
                            aligned2 = (homog2 @ np.array(m2).T)[:, :3]
                            if np.sqrt(np.mean(np.sum((aligned2 - tgt_pts2)**2, axis=1))) < shape_match_tol:
                                M2 = m2
                                break

                    if M1 and M2:
                        diff = np.linalg.norm(np.array(M1) - np.array(M2))
                        if diff < matrix_tol:
                            final_matrix = M1
                            cap['reference_mesh'] = primary_name
                            cap['validation_mesh'] = secondary_name
                            break
                        else:
                            consensus_failures.append(f"'{primary_name}' vs '{secondary_name}' (diff: {diff:.3f})")

            # 5) Apply or report skip
            if final_matrix:
                print(f"SUCCESS: Aligned using '{cap['reference_mesh']}' and '{cap['validation_mesh']}'.")
                for nm in cap['objects']:
                    ob = context.scene.objects.get(nm)
                    if is_ninja(ob):
                        if "ninjafix_prev_matrix" not in ob:
                            ob["ninjafix_prev_matrix"] = mat_to_list(ob.matrix_world)
                        ob.matrix_world = final_matrix @ ob.matrix_world
                successful_anchors.extend([cap['reference_mesh'], cap['validation_mesh']])
                for h in geom_cache.values():
                    SEEN_GEOM_HASHES.add(h)
                ph = tex_hash_cache[cap['reference_mesh']]
                if ph:
                    PREFERRED_ANCHOR_HASHES.add(ph)
                aligned_count += 1
            else:
                print(f"SKIPPED: Could not find a valid, verified anchor pair for '{cap['name']}'.")
                print("-" * 40)
                for reason, meshes in sorted(failure_reasons.items()):
                    if meshes:
                        if reason == "Secondary: Too Close to Primary":
                            print(f"- {reason}: {len(meshes)} candidates rejected.")
                        else:
                            print(f"- {reason}: {', '.join(sorted(meshes))}")
                if consensus_failures:
                    print("- Consensus Failed Between Pairs:")
                    for info in consensus_failures[:3]:
                        print(f"    - {info}")
                    if len(consensus_failures) > 3:
                        print(f"    - ... and {len(consensus_failures) - 3} others")
                skipped_count += 1

        # 6) Final selection and write-back
        if successful_anchors:
            import bpy
            bpy.ops.object.select_all(action='DESELECT')
            for name in set(successful_anchors):
                obj = context.scene.objects.get(name)
                if obj:
                    obj.select_set(True)
            bpy.context.view_layer.objects.active = context.scene.objects.get(successful_anchors[0])

        import json
        context.scene["ninjafix_capture_sets"] = json.dumps([
            {
                'name': c['name'],
                'objects': list(c['objects']),
                'reference_mesh': c.get('reference_mesh',''),
                'validation_mesh': c.get('validation_mesh','')
            }
            for c in CAPTURE_SETS
        ])

        self.report({'INFO'}, f"Processing complete: {aligned_count} aligned, {skipped_count} skipped.")
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
