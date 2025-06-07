#
# NinjaFix Aligner – FULL ADD-ON v3.1.6
# Date: 2025-06-07
#
# Aligns distorted Ninja Ripper meshes to a correct RTX Remix mesh
# using a guided, step-by-step vertex confirmation system.
# Supports full affine transforms (non-uniform scale, shear) via least squares.
#

bl_info = {
    "name":        "NinjaFix Aligner",
    "author":      "Gemini/Google & User",
    "version":     (3, 1, 6),
    "blender":     (3, 0, 0),
    "category":    "Object",
    "description": "Fixes Ninja Ripper meshes using a Remix reference and a guided vertex workflow.",
    "warning":     "Requires one-time internet to auto-install NumPy.",
}

import bpy
import bmesh
import subprocess
import sys
from mathutils import Vector, Matrix
from bpy.props   import PointerProperty, FloatVectorProperty, BoolProperty, StringProperty
from bpy.types   import PropertyGroup, Operator, Panel

# NumPy dependency
try:
    import numpy as np
    NUMPY_INSTALLED = True
except ImportError:
    np = None
    NUMPY_INSTALLED = False

# --- Property Group ---
class NinjaFixSettings(PropertyGroup):
    source_obj: PointerProperty(type=bpy.types.Object, name="Source (Ninja)")
    target_obj: PointerProperty(type=bpy.types.Object, name="Reference (Remix)")

    remix_v1: FloatVectorProperty(name="Remix Vertex 1"); remix_v1_picked: BoolProperty(default=False)
    ninja_v1: FloatVectorProperty(name="Ninja Vertex 1"); ninja_v1_picked: BoolProperty(default=False)
    remix_v2: FloatVectorProperty(name="Remix Vertex 2"); remix_v2_picked: BoolProperty(default=False)
    ninja_v2: FloatVectorProperty(name="Ninja Vertex 2"); ninja_v2_picked: BoolProperty(default=False)
    remix_v3: FloatVectorProperty(name="Remix Vertex 3"); remix_v3_picked: BoolProperty(default=False)
    ninja_v3: FloatVectorProperty(name="Ninja Vertex 3"); ninja_v3_picked: BoolProperty(default=False)
    remix_v4: FloatVectorProperty(name="Remix Vertex 4"); remix_v4_picked: BoolProperty(default=False)
    ninja_v4: FloatVectorProperty(name="Ninja Vertex 4"); ninja_v4_picked: BoolProperty(default=False)

    stored_transform: FloatVectorProperty(name="Stored Transformation Matrix", size=16, default=[1]*16)
    is_transform_stored: BoolProperty(default=False)

    def are_all_points_picked(self) -> bool:
        return all(getattr(self, f"{p}_picked") for p in (
            "remix_v1","ninja_v1",
            "remix_v2","ninja_v2",
            "remix_v3","ninja_v3",
            "remix_v4","ninja_v4"
        ))

# --- Operators ---

class NFIX_OT_InstallNumpy(Operator):
    bl_idname = "nfix.install_numpy"
    bl_label  = "Install NumPy"
    def execute(self, context):
        self.report({'INFO'}, "Installing NumPy… please wait.")
        python_exe = sys.executable
        try:
            subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call([python_exe, "-m", "pip", "install", "numpy"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.report({'INFO'}, "NumPy installed successfully! Restart Blender.")
        except Exception:
            self.report({'ERROR'}, "NumPy installation failed; check console.")
        return {'FINISHED'}

class NFIX_OT_SetObjects(Operator):
    bl_idname = "nfix.set_objects"
    bl_label  = "Set Ninja & Remix Objects"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.selected_objects) == 2 and context.mode == 'OBJECT'

    def execute(self, context):
        src = tgt = None

        # Prefix‐based selection
        for ob in context.selected_objects:
            if ob.name.startswith("mesh_"):
                src = ob
            elif ob.name.startswith("mesh."):
                tgt = ob

        # Fallback
        if not src or not tgt:
            for ob in context.selected_objects:
                if not src and "_" in ob.name:
                    src = ob
                if not tgt and "." in ob.name:
                    tgt = ob

        if not (src and tgt):
            self.report({'ERROR'},
                "Name one mesh as ‘mesh_xxx’ (Ninja) and one as ‘mesh.xxx’ (Remix).")
            return {'CANCELLED'}

        st = context.scene.ninjafix_settings
        st.source_obj = src
        st.target_obj = tgt

        # Keep both selected; make Remix active and enter Edit Mode
        context.view_layer.objects.active = tgt
        bpy.ops.object.mode_set(mode='EDIT')

        # <<< FIX: immediately deselect all vertices >>> 
        bpy.ops.mesh.select_all(action='DESELECT')

        bpy.ops.nfix.clear_all_points()
        self.report({'INFO'},
            f"Source = '{src.name}', Target = '{tgt.name}'. Ready to pick vertices.")
        return {'FINISHED'}

class NFIX_OT_ConfirmVertex(Operator):
    bl_idname = "nfix.confirm_vertex"
    bl_label  = "Pick Vertex"
    target_prop_name: StringProperty()

    @classmethod
    def poll(cls, context):
        return context.mode.startswith("EDIT") and context.edit_object is not None

    def execute(self, context):
        st = context.scene.ninjafix_settings
        active = context.edit_object
        is_remix = self.target_prop_name.startswith("remix")
        expected = st.target_obj if is_remix else st.source_obj

        if active != expected:
            self.report({'ERROR'}, f"Switch Edit Mode to '{expected.name}' first.")
            return {'CANCELLED'}

        bm = bmesh.from_edit_mesh(active.data)
        verts = [v for v in bm.verts if v.select]
        if len(verts) != 1:
            self.report({'ERROR'}, "Select exactly ONE vertex, then click the button.")
            return {'CANCELLED'}

        world_co = active.matrix_world @ verts[0].co
        setattr(st, self.target_prop_name, world_co)
        setattr(st, self.target_prop_name + "_picked", True)
        self.report({'INFO'}, f"{self.target_prop_name.replace('_',' ').title()} recorded.")
        return {'FINISHED'}

class NFIX_OT_ClearSinglePoint(Operator):
    bl_idname = "nfix.clear_single_point"
    bl_label  = "Clear Point"
    bl_options = {'REGISTER', 'UNDO'}
    target_prop_name: StringProperty()

    def execute(self, context):
        st = context.scene.ninjafix_settings
        setattr(st, self.target_prop_name, (0,0,0))
        setattr(st, self.target_prop_name + "_picked", False)
        st.is_transform_stored = False
        return {'FINISHED'}

class NFIX_OT_ClearAllPoints(Operator):
    bl_idname = "nfix.clear_all_points"
    bl_label  = "Clear All Points"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        st = context.scene.ninjafix_settings
        for i in range(1,5):
            for prefix in ("remix","ninja"):
                setattr(st, f"{prefix}_v{i}", (0,0,0))
                setattr(st, f"{prefix}_v{i}_picked", False)
        st.is_transform_stored = False
        self.report({'INFO'}, "All picks cleared.")
        return {'FINISHED'}

class NFIX_OT_CalculateAndBatchFix(Operator):
    bl_idname = "nfix.calculate_and_batch_fix"
    bl_label  = "Calculate & Apply to All"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return NUMPY_INSTALLED and context.scene.ninjafix_settings.are_all_points_picked()

    def execute(self, context):
        # exit Edit Mode
        if context.mode.startswith("EDIT"):
            bpy.ops.object.mode_set(mode='OBJECT')

        st = context.scene.ninjafix_settings

        # Source and target points
        src = np.array([st.ninja_v1, st.ninja_v2, st.ninja_v3, st.ninja_v4])
        tgt = np.array([st.remix_v1, st.remix_v2, st.remix_v3, st.remix_v4])

        # Augment with ones for affine solve
        ones = np.ones((4,1))
        X_aug = np.hstack((src, ones))  # shape (4,4)

        # Least squares solve X_aug @ A = tgt  →  A is 4×3
        A, *_ = np.linalg.lstsq(X_aug, tgt, rcond=None)

        # Build 4×4 matrix
        M = Matrix.Identity(4)
        for i in range(3):       # target coord index
            for j in range(4):   # source coord + bias
                M[i][j] = A[j, i]

        # Apply to primary
        st.source_obj.matrix_world = M
        st.stored_transform = [v for row in M for v in row]
        st.is_transform_stored = True
        self.report({'INFO'}, "Primary mesh aligned with full affine transform.")

        # Batch apply
        original = st.source_obj.name
        count = 0
        for ob in context.scene.objects:
            if ob.type == 'MESH' and '_' in ob.name and ob.name != original:
                ob.matrix_world = M
                count += 1
        self.report({'INFO'}, f"Applied transform to {count} other Ninja meshes.")
        return {'FINISHED'}

# --- UI Panel ---
class NFIX_OT_DeleteDuplicateNinjas(Operator):
    bl_idname = "nfix.delete_duplicate_ninjas"
    bl_label  = "Delete Duplicate Ninjas"
    bl_description = "Remove any Ninja mesh whose vertices approximately match a Remix mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tol = 0.01
        scene = context.scene
        deleted = 0

        # Gather world-space coords for each Remix mesh
        remix_sets = []
        for ob in scene.objects:
            if ob.type == 'MESH' and '.' in ob.name and '_' not in ob.name:
                coords = [ob.matrix_world @ v.co for v in ob.data.vertices]
                remix_sets.append(coords)

        # For each Ninja mesh, compare against each Remix set
        for ob in list(scene.objects):
            if ob.type == 'MESH' and '_' in ob.name and '.' not in ob.name:
                ninja_coords = [ob.matrix_world @ v.co for v in ob.data.vertices]
                for rv in remix_sets:
                    if len(rv) == len(ninja_coords):
                        # compute RMSE
                        sq_diffs = [(ninja_coords[i] - rv[i]).length_squared for i in range(len(rv))]
                        rmse = (sum(sq_diffs) / len(sq_diffs))**0.5
                        if rmse < tol:
                            bpy.data.objects.remove(ob, do_unlink=True)
                            deleted += 1
                            break

        self.report({'INFO'}, f"Deleted {deleted} duplicate Ninja mesh(es) (tol={tol}).")

        # ← Insert here:
        # Count how many Ninja meshes remain
        remaining = sum(1 for ob in scene.objects
                        if ob.type == 'MESH' and '_' in ob.name and '.' not in ob.name)
        print(f"[NinjaFix] {remaining} ninja mesh(es) remain in the scene.")

        return {'FINISHED'}

class NFIX_OT_ClearAllSelections(Operator):
    bl_idname = "nfix.clear_all_selections"
    bl_label  = "Clear All Selections"
    bl_description = "Deselect the Ninja/Remix meshes and clear all picked vertices"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        st = context.scene.ninjafix_settings

        # Clear mesh selections
        st.source_obj = None
        st.target_obj = None

        # Clear all vertex picks
        bpy.ops.nfix.clear_all_points()

        self.report({'INFO'}, "Cleared mesh selections and all vertex picks.")
        return {'FINISHED'}

class VIEW3D_PT_NinjaFix_Aligner(Panel):
    bl_label       = "NinjaFix Aligner"
    bl_idname      = "VIEW3D_PT_nfix_aligner"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "NinjaFix"

    def draw(self, context):
        layout = self.layout
        st     = context.scene.ninjafix_settings

        # Dependency
        if not NUMPY_INSTALLED:
            box = layout.box()
            box.label(text="NumPy required", icon='ERROR')
            box.operator(NFIX_OT_InstallNumpy.bl_idname, icon='CONSOLE')
            return

        # Stage 1: Setup
        box1 = layout.box()
        box1.label(text="Stage 1: Setup", icon='OBJECT_DATA')
        col  = box1.column()
        col.enabled = (context.mode == 'OBJECT')
        col.label(text="Select one Ninja and one Remix mesh, then:")
        col.operator(NFIX_OT_SetObjects.bl_idname, icon='CHECKMARK')
        # Clear button beneath
        col.operator(NFIX_OT_ClearAllSelections.bl_idname, icon='X', text="Clear All Selections")
        if st.source_obj and st.target_obj:
            box1.label(text=f"Ninja Source: {st.source_obj.name}", icon='MESH_CUBE')
            box1.label(text=f"Remix Target: {st.target_obj.name}", icon='MESH_UVSPHERE')

        # Stage 2: Pick Vertices
        box2 = layout.box()
        box2.label(text="Stage 2: Pick Vertices", icon='RESTRICT_SELECT_OFF')

        sequence = [f"{p}_v{i}" for i in range(1,5) for p in ("remix","ninja")]
        next_prop = next((p for p in sequence if not getattr(st, p + "_picked")), None)

        # show completed with clear buttons
        for p in sequence:
            if getattr(st, p + "_picked"):
                row = box2.row(align=True)
                row.label(text=f"✓ {p.replace('_',' ').title()}", icon='CHECKMARK')
                clr = row.operator(NFIX_OT_ClearSinglePoint.bl_idname, text="", icon='X')
                clr.target_prop_name = p

        # pick button for next
        if next_prop:
            box2.separator()
            row = box2.row()
            row.enabled = context.mode.startswith("EDIT")
            op = row.operator(NFIX_OT_ConfirmVertex.bl_idname,
                              text=f"Pick {next_prop.replace('_',' ').title()}",
                              icon='MESH_DATA')
            op.target_prop_name = next_prop
            if not context.mode.startswith("EDIT"):
                box2.label(text="(Enter Edit Mode on Ninja or Remix mesh)", icon='INFO')
        else:
            box2.label(text="All vertices picked ✓", icon='CHECKMARK')

        # Stage 3: Calculate & Apply
        box3 = layout.box()
        box3.enabled = st.are_all_points_picked()
        box3.label(text="Stage 3: Apply Transformation", icon='PLAY')
        row = box3.row()
        row.scale_y = 1.5
        row.operator(NFIX_OT_CalculateAndBatchFix.bl_idname, text="Generate & Apply Fix")

        # <— NEW: delete duplicates button appears once transform is applied
        del_row = box3.row()
        del_row.enabled = True
        del_row.operator(NFIX_OT_DeleteDuplicateNinjas.bl_idname, icon='TRASH')

# --- Registration ---

classes = (
    NinjaFixSettings,
    NFIX_OT_InstallNumpy,
    NFIX_OT_SetObjects,
    NFIX_OT_ClearAllSelections,      # ← newly added
    NFIX_OT_ConfirmVertex,
    NFIX_OT_ClearSinglePoint,
    NFIX_OT_ClearAllPoints,
    NFIX_OT_CalculateAndBatchFix,
    NFIX_OT_DeleteDuplicateNinjas,
    VIEW3D_PT_NinjaFix_Aligner,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ninjafix_settings = PointerProperty(type=NinjaFixSettings)

def unregister():
    del bpy.types.Scene.ninjafix_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
