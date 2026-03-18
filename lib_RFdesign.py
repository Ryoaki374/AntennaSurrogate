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

class ConvexBackshort(Convex):
    """
    Specific class for Backshort generation, inheriting general Convex tools.
    """
    def genBackshort(self, a=9.525, b=4.7625, c=-7.725, k=6, grid_res=30, shifts=(0, -4.7625, -0.34575)):
        """
        Generates the backshort geometry and exports it to STEP.
        Maintains original mathematical formulas and function names.
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
        
        # Secure export using Pathlib joined path
        exporters.export(result, str(self.model_path))
        
        return hull_data
    

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

    
