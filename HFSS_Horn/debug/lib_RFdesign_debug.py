"""Minimal helpers for creating and displaying a solid cylinder."""

from pathlib import Path

import cadquery as cq
import matplotlib.pyplot as plt
import numpy as np
from cadquery import exporters
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Cylinder:
    """Create a circular cylinder or an inscribed regular polygonal prism."""

    def __init__(self, model_path: Path):
        if model_path is None:
            raise ValueError("model_path cannot be None.")
        self.model_path = Path(model_path)

    def genCylinder(self, diameter=10.0, height=20.0, n_vertices=None,
                    circle_plot_vertices=72):
        """Generate a filled solid and export it to ``model_path`` as STEP.

        Parameters are in millimetres.  With ``n_vertices=None`` the section
        is an exact circle.  Otherwise it is a regular polygon whose vertices
        lie on that circle (and is therefore inscribed in it).

        Returns
        -------
        points : (2*N, 3) ndarray
            Bottom-ring points followed by top-ring points, for
            :meth:`plotCylinder3D`.
        """
        diameter = float(diameter)
        height = float(height)
        if diameter <= 0.0 or height <= 0.0:
            raise ValueError("diameter and height must be positive.")

        if n_vertices is None:
            n_plot = int(circle_plot_vertices)
            if n_plot < 3:
                raise ValueError("circle_plot_vertices must be at least 3.")
            solid = cq.Workplane("XY").circle(diameter / 2.0).extrude(height)
        else:
            if isinstance(n_vertices, (bool, np.bool_)) or not isinstance(
                    n_vertices, (int, np.integer)) or n_vertices < 3:
                raise ValueError("n_vertices must be None or an integer >= 3.")
            n_plot = int(n_vertices)
            section = self._section_points(diameter / 2.0, n_plot)
            solid = (cq.Workplane("XY")
                     .polyline([tuple(point) for point in section])
                     .close()
                     .extrude(height))

        exporters.export(solid, str(self.model_path))
        section = self._section_points(diameter / 2.0, n_plot)
        bottom = np.column_stack((section, np.zeros(n_plot)))
        top = np.column_stack((section, np.full(n_plot, height)))
        return np.vstack((bottom, top))

    @staticmethod
    def _section_points(radius, n_vertices):
        theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
        return np.column_stack((radius * np.cos(theta),
                                radius * np.sin(theta)))

    def plotCylinder3D(self, points):
        """Display points returned by :meth:`genCylinder` in 3D."""
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3
                or len(points) < 6 or len(points) % 2):
            raise ValueError("points must have shape (2*N, 3), with N >= 3.")
        n_vertices = len(points) // 2
        bottom, top = points[:n_vertices], points[n_vertices:]

        faces = [bottom[::-1], top]
        for i in range(n_vertices):
            j = (i + 1) % n_vertices
            faces.append([bottom[i], bottom[j], top[j], top[i]])

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_edgecolor("k")
            axis.pane.set_facecolor("w")
        ax.grid(False)
        ax.view_init(azim=50, elev=30)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   color="k", s=10, alpha=0.5)
        ax.add_collection3d(Poly3DCollection(
            faces, alpha=0.3, facecolors="gray", edgecolors="k"))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect(np.ptp(points, axis=0))
        plt.show()
        return fig, ax
