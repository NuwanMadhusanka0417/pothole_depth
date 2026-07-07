# Pothole depth from RGB + IMU — improvement plan

## The problem with the current estimate

The `imu` pipeline currently derives depth from the **bounding box** with the
trigonometric model in `geometric_depth.py`:

```
d_top    = h / tan(theta_top)       # bbox top edge  -> far rim of the hole
d_bottom = h / tan(theta_bottom)    # bbox bottom edge -> near rim of the hole
depth    = (d_top - d_bottom) * tan(theta_top)
```

Both the top and bottom edges of the box lie on the **road surface** (the rim
of the opening). So `d_top - d_bottom` is the pothole's *footprint length along
the road* (~0.7-1 m), and `x tan(22 deg) ~ 0.4` yields ~30-40 cm. It measures
the ground footprint, **not** the vertical drop. That is why a 5-7 cm pothole
reads as 30-40 cm. This is a *method* error, not a calibration error, and
cannot be tuned away. A 2D box contains no information about how far the floor
drops below the road.

**Fix:** measure depth as the perpendicular drop of the pothole floor below a
fitted road plane, from a metric 3D representation of the pothole region. Two
independent ways to get that 3D, both RGB+IMU only:

---

## Path A — Physics-based: SfM/MVS + IMU scale (primary)

Reconstruct the actual 3D surface of the pothole (including its floor) by
triangulating across frames as the camera moves, then measure the floor's drop
below the road plane.

**Steps**
1. YOLO detect + track the pothole (existing).
2. Multi-view reconstruction of the pothole region -> sparse/dense 3D point
   cloud (`reconstruction/sfm_runner.py`).
3. **Metric scale from IMU**: known camera height + gravity give absolute scale
   (`reconstruction/scale_recovery.py`); inertial scale recovery is ~1% error in
   the literature.
4. **Road plane** from the IMU gravity vector + RANSAC fit to the rim/road ring.
5. **Depth = max perpendicular drop of interior points below the plane**
   (`geometry/plane_depth.py`, shared core).
6. Median / Kalman smoothing across frames.

**Why primary:** it is physically grounded and, per the literature, resolves
shallow potholes to ~1 cm at close range (Springer Applied Geomatics 2025;
Aston SfM pothole reconstruction). It reuses modules already in the repo.

**Dependencies / status:** needs COLMAP (or the built-in OpenCV fallback in
`sfm_runner.py`). Not yet wired into the `imu` subcommand (that path is
geometric-only today) — see "Code changes".

---

## Path B — Learned: monocular metric depth + gravity plane fit (validator)

Run a pretrained **metric** monocular depth network on the frame, back-project
to a metric point cloud, and measure the same below-plane drop.

**Steps**
1. YOLO detect + track (existing).
2. **DepthAnything V2 (metric)** -> per-pixel metric depth map
   (`fusion/mono_depth.py`).
3. Back-project the pothole ROI to 3D using the (upright) intrinsics.
4. **Road plane** from IMU gravity + RANSAC on the road ring
   (`geometry/plane_depth.py`, shared core).
5. **Depth = max drop of interior points below the plane.**
6. Median / Kalman smoothing across frames.

**Why validator:** it is single-frame, fast, needs no camera motion, and does
not assume a flat road. It is the modern SOTA baseline (arXiv 2505.21049 uses
DepthAnything V2 + detection + 3D mapping). Caveat: learned depth over-smooths,
so a 5 cm dip at 1-2 m may be under-resolved — B is better for confirming
*presence and rough magnitude* than for the last centimetre.

**Dependencies / status:** `pip install transformers timm` (torch already
present). `mono_depth.py` loads the model lazily and errors with the install
command if missing.

---

## Testing Path A using Path B

Path B is the **independent cross-check** for the physics-based Path A. Both
consume the *same* `geometry/plane_depth.py` core, so any disagreement is due to
the 3D source (SfM vs learned depth), not the depth-extraction math.

**Procedure**
1. Run A and B on the same recording; per tracked pothole get `depth_A`, `depth_B`.
2. **Agreement metrics** (`fusion/depth_compare.py`):
   - absolute difference `|depth_A - depth_B|`
   - relative gap `|A-B| / mean(A,B)`
   - across a dataset: MAE, RMSE, bias (mean A-B), Pearson correlation.
3. **Interpretation**
   - Small gap (< ~1 cm or < ~15%) -> A validated; publish A with high confidence.
   - Large gap -> flag; inspect (bad SfM scale, too few 3D points, over-smoothed B).
4. **Ground truth anchor**: for a subset, measure real depth with a ruler.
   Report A-vs-GT and B-vs-GT so the A/B agreement is calibrated against truth
   (agreement alone only shows consistency, not correctness).
5. The consensus/fusion in `fusion/hybrid_estimator.py` already turns A/B
   agreement into a confidence score — reuse it.

**Success criteria**
- A vs B relative gap < 15% median on the test clips.
- A vs ruler GT MAE <= ~1.5 cm on the close-range validation subset.

---

## Code changes (this repo)

Shared core (new):
- `geometry/plane_depth.py` — road-plane fit (gravity-oriented RANSAC) +
  below-plane depression measurement. Source-agnostic: takes any metric depth
  map or 3D points. Used by BOTH paths. **Unit-tested on a synthetic pothole.**
- `fusion/mono_depth.py` — DepthAnything V2 metric wrapper (Path B backend).
- `fusion/depth_compare.py` — A-vs-B agreement metrics.
- `geometry/imu_orientation.py::gravity_camera_frame` — device gravity ->
  upright OpenCV camera frame (needed to orient the plane).

Wiring:
- `main.py` `imu` subcommand gains `--depth-method {geometric,monodepth,both}`.
  `monodepth` = Path B; `both` computes B and the old geometric bound and
  records the comparison in `summary.json`.
- Path A (SfM) into the `imu` command and `hybrid_estimator` switching its crude
  Y-spread depth to `plane_depth` are the next step (needs COLMAP); tracked here.

## Honest accuracy expectation
A 5 cm pothole at 1-2 m is near the monocular noise floor. Expect a few cm of
uncertainty; capture close + slow, and always validate against ruler ground
truth before quoting an accuracy figure.
