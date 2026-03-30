import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from cqmore import Workplane
import cadquery as cq
from cadquery import exporters
from pathlib import Path

class Convex:
    def __init__(self, model_path: Path):
        """
        Base class for Convex Hull operations.
        run_dir: Pathlib object from the backbone instance.
        """
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            raise ValueError("model_path cannot be None. A valid directory Path is required.")  

    def plotConvex3D(self, hull):
        """
        Visualizes the convex hull in 3D.
        Maintains the original visual logic and aspect ratio setting.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Configure 3D pane appearance
        ax.xaxis.pane.set_edgecolor('k')
        ax.yaxis.pane.set_edgecolor('k')
        ax.zaxis.pane.set_edgecolor('k')
        ax.xaxis.pane.set_facecolor("w")
        ax.yaxis.pane.set_facecolor("w")
        ax.zaxis.pane.set_facecolor("w")

        ax.grid(False)
        ax.view_init(azim=50, elev=30)

        # 1. Plot point cloud
        pts = hull.points
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='k', s=10, alpha=0.5)

        # 2. Draw faces
        faces = [pts[s] for s in hull.simplices]
        poly = Poly3DCollection(faces, alpha=0.3, facecolors='gray', edgecolors='k')
        ax.add_collection3d(poly)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Maintains the original aspect ratio command
        ax.set_aspect('equal')
        
        plt.show()

    def plotProfile2D(self, curve_pts,):

        curve_pts = np.asarray(curve_pts, dtype=float)

        pts = curve_pts

        fig, ax = plt.subplots(figsize=(7, 6))

        # Plot the polyline (ordered)
        ax.plot(pts[:, 0], pts[:, 1], lw=1.5, c="k")

        # Explicitly close the loop if close_pts is provided
        ax.plot([pts[-1, 0], pts[0, 0]], [pts[-1, 1], pts[0, 1]], lw=1.5, c="k")

        # Optionally show points for debugging
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c="k", alpha=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, c='k')

        plt.show()

    def plotStepBackshort3D(self, step_info):
        """Visualizes a step-backshort metadata dict returned by genStepBackshort."""
        boxes = step_info.get('boxes', [])
        if not boxes:
            raise ValueError('step_info must contain a non-empty boxes list.')

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.pane.set_edgecolor('k')
        ax.yaxis.pane.set_edgecolor('k')
        ax.zaxis.pane.set_edgecolor('k')
        ax.xaxis.pane.set_facecolor('w')
        ax.yaxis.pane.set_facecolor('w')
        ax.zaxis.pane.set_facecolor('w')
        ax.grid(False)
        ax.view_init(azim=50, elev=30)

        all_vertices = []
        for box in boxes:
            x0, x1 = box['x_min'], box['x_max']
            y0, y1 = box['y_min'], box['y_max']
            z0, z1 = box['z_min'], box['z_max']
            vertices = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ])
            all_vertices.append(vertices)
            faces = [
                vertices[[0, 1, 2, 3]],
                vertices[[4, 5, 6, 7]],
                vertices[[0, 1, 5, 4]],
                vertices[[1, 2, 6, 5]],
                vertices[[2, 3, 7, 6]],
                vertices[[3, 0, 4, 7]],
            ]
            poly = Poly3DCollection(faces, alpha=0.25, facecolors='gray', edgecolors='k')
            ax.add_collection3d(poly)
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='k', s=10, alpha=0.4)

        pts = np.vstack(all_vertices)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
        ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
        ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())
        ax.set_aspect('equal')
        plt.show()

class ConvexBackshort(Convex):
    """
    Specific class for Backshort generation, inheriting general Convex tools.
    """
    def genBackshort(self, a=9.525, b=4.7625, c=-7.725, k=6, grid_res=30, shifts=(0, -4.7625, -0.34575)):
        """
        Generates the original smooth backshort geometry and exports it to STEP.
        Maintains the original mathematical formulas and function names.
        """
        shift_x, shift_y, shift_z = shifts

        # 1. Generate point cloud
        x = np.linspace(-a, a, grid_res)
        y = np.linspace(-b, b, grid_res)
        X, Y = np.meshgrid(x, y)

        # Super-ellipsoid style surface calculation logic
        Z = c * np.sqrt(np.maximum(0, 1 - (X / a)**(2*k)) * np.maximum(0, 1 - (Y / b)**(2*k)))

        # 2. Apply translations
        X += shift_x
        Y += shift_y
        Z += shift_z

        # Create coordinate array
        raw_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # 3. Compute Convex Hull
        hull_data = ConvexHull(raw_points)

        # 4. Create CadQuery solid and export
        result = Workplane().polyhedron(hull_data.points, hull_data.simplices)

        exporters.export(result, str(self.model_path))
        return hull_data

    def genStepBackshort(
        self,
        a=9.525,
        b=4.7625,
        step_heights=(2.0, 2.0, 2.0),
        shrink=1.5,
        shifts=(0, -4.7625, -0.34575),
    ):
        """
        Generates a negative-Z step-backshort by stacking shrinking boxes.

        Parameters
        ----------
        a, b : float
            Base half-widths in X/Y.
        step_heights : sequence[float]
            Per-step thickness values. Each entry is stacked along negative Z.
        shrink : float
            XY shrink factor applied automatically for each higher step.
        shifts : tuple[float, float, float]
            Final translation applied to the stacked solid.
        """
        shift_x, shift_y, shift_z = shifts
        heights = [float(h) for h in step_heights if float(h) > 0]
        if not heights:
            raise ValueError('step_heights must contain at least one positive thickness.')
        if shrink <= 1.0:
            raise ValueError('shrink must be greater than 1.0.')

        solid = None
        z_cursor = 0.0
        for i, height in enumerate(heights):
            half_x = float(a) / (shrink ** i)
            half_y = float(b) / (shrink ** i)
            width_x = 2.0 * half_x
            width_y = 2.0 * half_y
            z_min = -(z_cursor + height)

            box = (
                cq.Workplane('XY')
                .box(width_x, width_y, height, centered=(True, True, False))
                .translate((0.0, 0.0, z_min))
            )
            solid = box if solid is None else solid.union(box)
            z_cursor += height

        boxes = []
        z_cursor = 0.0
        for i, height in enumerate(heights):
            half_x = float(a) / (shrink ** i)
            half_y = float(b) / (shrink ** i)
            z_min = -(z_cursor + height)
            boxes.append({
                'x_min': -half_x + shift_x,
                'x_max': half_x + shift_x,
                'y_min': -half_y + shift_y,
                'y_max': half_y + shift_y,
                'z_min': z_min + shift_z,
                'z_max': z_min + height + shift_z,
            })
            z_cursor += height

        solid = solid.translate((shift_x, shift_y, shift_z))
        exporters.export(solid, str(self.model_path))
        return {
            'type': 'stepbackshort',
            'base_half_width': (float(a), float(b)),
            'step_heights': heights,
            'n_steps': len(heights),
            'shrink': float(shrink),
            'total_depth': float(sum(heights)),
            'boxes': boxes,
        }


    def genStepBackshortCont(
        self,
        a=9.525,
        b=4.7625,
        step_heights=(2.0, 2.0, 2.0, 2.0, 2.0),
        shrink_params=(1.0, 0.2, 0.2, 0.2, 0.2),
        shifts=(0, -4.7625, -0.34575),
    ):
        """
        Generates a 5-step negative-Z step-backshort with monotonic per-step XY shrink factors.

        Parameters
        ----------
        a, b : float
            Base half-widths in X/Y.
        step_heights : sequence[float]
            Per-step thickness values (5 steps). Each entry is stacked along negative Z.
        shrink_params : sequence[float]
            Re-parameterized shrink controls used to guarantee monotonic ordering.
            The tuple must be (s1, s2, s3, s4, s5), where all values > 0.
            Per-step shrink factors are reconstructed cumulatively as:
              shrink_1 = s1
              shrink_2 = s1 + s2
              shrink_3 = s1 + s2 + s3
              shrink_4 = s1 + s2 + s3 + s4
              shrink_5 = s1 + s2 + s3 + s4 + s5
            This guarantees shrink_1 < shrink_2 < shrink_3 < shrink_4 < shrink_5.
        shifts : tuple[float, float, float]
            Final translation applied to the stacked solid.
        """
        shift_x, shift_y, shift_z = shifts

        heights = [float(h) for h in step_heights]
        sp = [float(v) for v in shrink_params]

        if len(heights) != 5:
            raise ValueError('step_heights must contain exactly 5 values.')
        if len(sp) != 5:
            raise ValueError('shrink_params must contain exactly 5 values: (s1, s2, s3, s4, s5).')
        if any(h <= 0.0 for h in heights):
            raise ValueError('All step_heights values must be positive.')

        s1, s2, s3, s4, s5 = sp
        if any(v <= 0.0 for v in (s1, s2, s3, s4, s5)):
            raise ValueError('s1..s5 must be positive.')

        shrink_vals = [
            s1,
            s1 + s2,
            s1 + s2 + s3,
            s1 + s2 + s3 + s4,
            s1 + s2 + s3 + s4 + s5,
        ]

        solid = None
        z_cursor = 0.0
        for height, shrink in zip(heights, shrink_vals):
            half_x = float(a) / shrink
            half_y = float(b) / shrink
            width_x = 2.0 * half_x
            width_y = 2.0 * half_y
            z_min = -(z_cursor + height)

            box = (
                cq.Workplane('XY')
                .box(width_x, width_y, height, centered=(True, True, False))
                .translate((0.0, 0.0, z_min))
            )
            solid = box if solid is None else solid.union(box)
            z_cursor += height

        boxes = []
        z_cursor = 0.0
        for height, shrink in zip(heights, shrink_vals):
            half_x = float(a) / shrink
            half_y = float(b) / shrink
            z_min = -(z_cursor + height)
            boxes.append({
                'x_min': -half_x + shift_x,
                'x_max': half_x + shift_x,
                'y_min': -half_y + shift_y,
                'y_max': half_y + shift_y,
                'z_min': z_min + shift_z,
                'z_max': z_min + height + shift_z,
            })
            z_cursor += height

        solid = solid.translate((shift_x, shift_y, shift_z))
        exporters.export(solid, str(self.model_path))
        return {
            'type': 'stepbackshort_cont',
            'base_half_width': (float(a), float(b)),
            'step_heights': heights,
            'n_steps': len(heights),
            'shrink_params': sp,
            'shrinks': shrink_vals,
            'total_depth': float(sum(heights)),
            'boxes': boxes,
        }



class ConvexFinshape(Convex):
    
    def genFinshape(self, a=6, b=6, k=2, grid_res=400, shifts=(0.0, -1.0)):
      
        shift_x, shift_y = shifts
      
        t = np.linspace(0.0, np.pi, grid_res)

        x = a * np.sign(np.cos(t)) * (np.abs(np.cos(t)) ** (2.0 / k)) + shift_x
        y = b * (np.sign(np.sin(t)) * (np.abs(np.sin(t)) ** (2.0 / k)) - 1) + shift_y

        # Ordered boundary points on the arc
        curve_pts = list(zip(x, y))

        # -----------------------
        # Build a planar Face by closing the arc with a straight baseline
        wp = cq.Workplane("XY").polyline(curve_pts).close()

        # Turn the closed polyline into a wire, then create a planar face (sheet)
        wire = wp.wire().val()
        face = cq.Face.makeFromWires(wire)

        # export
        exporters.export(face, str(self.model_path))

        return curve_pts

    
