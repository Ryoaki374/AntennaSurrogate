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

class ConvexHorn(Convex):
    """
    Pinched conical horn with S-shaped (sigmoid) wall transitions.

    The profile, traced from the reference design, consists of five
    sections stacked along Z (bottom -> top):

        1. straight circular waveguide        (radius d_waveguide / 2)
        2. S-shaped taper  wg  -> middle
        3. straight intermediate "pinch"      (radius d_middle / 2)
        4. S-shaped taper  middle -> aperture
        5. straight aperture section          (radius d_aperture / 2)

    Each taper uses a sine-squared sigmoid,
        r(xi) = r1 + (r2 - r1) * sin(pi*xi/2)**2 ,
    whose slope vanishes at both ends, so every junction between a taper
    and a straight section is tangent-continuous (smooth, convex S-shape).
    """

    def genHorn(self,
                d_aperture=11.6,
                d_middle=6.4,
                d_waveguide=1.80,
                total_length=20.0,
                section_fracs=(0.18, 0.27, 0.21, 0.20, 0.14),
                n_pts=120):
        """
        Generates the horn solid (vacuum part) and exports it to STEP.

        Parameters
        ----------
        d_aperture : float
            Aperture (top) diameter [mm]; set by the pixel size.
        d_middle : float
            Diameter of the intermediate straight ("pinched") section [mm].
            The reference geometry corresponds to ~0.55 * d_aperture.
        d_waveguide : float
            Waveguide (bottom) diameter [mm]. For ~140 GHz (D band) a
            circular waveguide of D = 1.80 mm is used (TE11 cutoff ~97.6 GHz).
        total_length : float
            Total model length along Z [mm].
        section_fracs : 5-tuple of float, sums to 1
            Length fractions of the sections, bottom to top:
            (waveguide, taper1, middle, taper2, aperture).
            Defaults are traced from the reference image.
        n_pts : int
            Number of sample points per S-taper.

        Returns
        -------
        curve_pts : (N, 2) ndarray
            Closed (r, z) profile polyline, usable with plotProfile2D.
        """
        fr = np.asarray(section_fracs, dtype=float)
        if fr.size != 5 or np.any(fr <= 0):
            raise ValueError("section_fracs must be 5 positive fractions.")
        if not np.isclose(fr.sum(), 1.0):
            raise ValueError("section_fracs must sum to 1.")

        r_wg, r_mid, r_ap = d_waveguide/2.0, d_middle/2.0, d_aperture/2.0
        L = total_length
        # section boundaries along z
        zb = np.concatenate(([0.0], np.cumsum(fr))) * L

        def s_taper(z0, z1, r1, r2):
            """Sine-squared S-shaped taper from (z0, r1) to (z1, r2)."""
            xi = np.linspace(0.0, 1.0, n_pts)
            z = z0 + xi * (z1 - z0)
            r = r1 + (r2 - r1) * np.sin(0.5*np.pi*xi)**2
            return z, r

        # assemble outer wall, bottom -> top
        z_list = [0.0]
        r_list = [r_wg]                       # 1. waveguide bottom edge
        z_list.append(zb[1]); r_list.append(r_wg)

        z, r = s_taper(zb[1], zb[2], r_wg, r_mid)   # 2. taper wg -> middle
        z_list += list(z[1:]); r_list += list(r[1:])

        z_list.append(zb[3]); r_list.append(r_mid)  # 3. middle straight

        z, r = s_taper(zb[3], zb[4], r_mid, r_ap)   # 4. taper middle -> aperture
        z_list += list(z[1:]); r_list += list(r[1:])

        z_list.append(L); r_list.append(r_ap)       # 5. aperture straight

        # close the profile along the rotation axis (r = 0)
        r_closed = np.array([0.0] + r_list + [0.0])
        z_closed = np.array([0.0] + z_list + [L])
        curve_pts = np.column_stack((r_closed, z_closed))

        # revolve the closed profile around the global Z axis -> solid
        solid = (
            cq.Workplane("XZ")
            .polyline([tuple(p) for p in curve_pts])
            .close()
            .revolve(360.0, (0, 0, 0), (0, 1, 0))  # local Y of "XZ" = global Z
        )
        exporters.export(solid, str(self.model_path))

        return curve_pts

    def plotHorn3D(self, curve_pts, n_theta=72, step=1):
        """
        Visualizes the horn as a solid of revolution in 3D.

        The (r, z) profile returned by genHorn is revolved by one full turn
        around the profile's vertical axis (the y axis of plotProfile2D,
        i.e. the global Z axis), reproducing the exported solid. Implemented
        in the same visual style as Convex.plotConvex3D; a convex hull
        cannot be used here because the pinched profile is non-convex.

        Parameters
        ----------
        curve_pts : (N, 2) array
            Closed (r, z) profile polyline from genHorn.
        n_theta : int
            Number of angular samples for the revolution.
        step : int
            Profile decimation step for plotting (>=1). Note: decimation
            can drop sharp corner points; keep step=1 unless the profile
            is very dense and smooth.
        """
        pts2d = np.asarray(curve_pts, dtype=float)
        if step > 1:
            # decimate interior points but always keep the endpoints
            keep = np.zeros(len(pts2d), dtype=bool)
            keep[::step] = True
            keep[[0, 1, -2, -1]] = True
            pts2d = pts2d[keep]
        theta = np.linspace(0.0, 2.0*np.pi, n_theta + 1)

        # Revolve the profile: X = r cos(t), Y = r sin(t), Z = z
        R = pts2d[:, 0][:, None]
        X = R * np.cos(theta)[None, :]
        Y = R * np.sin(theta)[None, :]
        Z = np.repeat(pts2d[:, 1][:, None], n_theta + 1, axis=1)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Configure 3D pane appearance (same as plotConvex3D)
        ax.xaxis.pane.set_edgecolor('k')
        ax.yaxis.pane.set_edgecolor('k')
        ax.zaxis.pane.set_edgecolor('k')
        ax.xaxis.pane.set_facecolor("w")
        ax.yaxis.pane.set_facecolor("w")
        ax.zaxis.pane.set_facecolor("w")

        ax.grid(False)
        ax.view_init(azim=50, elev=30)

        # 1. Plot point cloud (decimated, as in plotConvex3D)
        ax.scatter(X[::6, ::4].ravel(), Y[::6, ::4].ravel(), Z[::6, ::4].ravel(),
                   color='k', s=10, alpha=0.5)

        # 2. Draw faces: quads between adjacent profile points and angles
        faces = []
        for i in range(X.shape[0] - 1):
            for j in range(n_theta):
                faces.append([(X[i, j],   Y[i, j],   Z[i, j]),
                              (X[i+1, j], Y[i+1, j], Z[i+1, j]),
                              (X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]),
                              (X[i, j+1], Y[i, j+1], Z[i, j+1])])
        poly = Poly3DCollection(faces, alpha=0.3, facecolors='gray',
                                edgecolors='k', linewidths=0.1)
        ax.add_collection3d(poly)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_aspect('equal')

        plt.show()
