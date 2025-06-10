## **NinjaFix Aligner (multi-capture)** 🚀

### **Perfect Alignment for Ninja Ripper & Remix**

Ninja Ripper is a tool that is able to rip meshes with shader transformations and import them into blender. However the meshes are scaled and rotated which makes them unusable. This Blender addon solves the problem by using a single remix mesh which is in both captures to find the correct transformations needed to fix the entire Ninja Ripper capture/s.

### ✨ **Key Features:**

* **Automatic Alignment:**
  Precisely align multiple Ninja Ripper captures to your Remix reference with minimal manual intervention.

* **Batch Processing:**
  Rapidly handle large numbers of meshes across multiple captures simultaneously, optimizing your workflow and saving valuable time.

* **Duplicate Detection & Removal:**
  Intelligently detect and safely remove duplicate meshes based on geometry, location, and material rules. Never lose important custom materials again.

* **Non-Uniform Affine Transforms:**
  Effortlessly manage and apply complex scaling, rotation, and translation transformations for perfect capture matches.

* **Efficient Caching & Speed Optimization:**
  Built-in caching of geometry evaluations, KD-trees, and texture hashes dramatically boosts performance for repetitive tasks.

* **Easy Capture Management:**
  Quickly add, remove, and track captures within Blender, streamlining large-scale asset workflows.

* **Persistent Storage:**
  Stores alignment settings directly alongside your `.blend` files for hassle-free future sessions.

* **Debugging and Verbose Logs:**
  Comprehensive debug output ensures transparency and easier troubleshooting during alignment and duplicate removal.

---

Efficient. Accurate. Powerful.

**How to Use**

Import your remix capture and one ninja capture. Remember to click Get Texcoords From Local Space in the Ninja Ripper Blender importer addon. Select two meshes which are the same in game and that do not have symmetrical geometry. Press Set from Selection, then Generate and Apply Fix. If you want to import more ninja captures, after each import, press Add New Capture. Then once you have imported all needed captures, press Process All Captures. If you would like to import more captures anytime after this, remove the captures from the list, then Force Capture will show, import your new capture/s while also pressing add new capture after each, then Process Call Captures.

The Ninja Ripper Blender addon can be found in C:\Program Files (x86)\Ninja Ripper 2.8\importers if you are using version 2.8. Adjust if needed. 

**NinjaFix Aligner**: Unlock your Blender productivity now!

![image](https://github.com/user-attachments/assets/91086cf8-0778-4d77-96ee-ad6a7fb75307)

