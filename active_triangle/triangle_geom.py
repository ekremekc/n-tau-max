import gmsh
import os 
if not os.path.exists('MeshDir'):
    os.makedirs('MeshDir')
from math import sqrt

def geometry(file="MeshDir/triangle", fltk=False):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(__name__)

    lc = 4e-3 # 3e-3 

    scaler = 1
    L = 1387*scaler  * 1e-3 #mm length of combustor default:1387mm
    H = 152.4*scaler * 1e-3 #mm height of combustor default:152.4mm
    W = 127   * 1e-3 #mm width  of combustor default:127mm

    flame_loc = 864 * scaler * 1e-3 # mm
    D = 38.1 * scaler * 1e-3 #mm
    geom = gmsh.model.geo

    p1 = geom.addPoint(0, -H/2, 0, lc)
    p2 = geom.addPoint(L, -H/2, 0, lc)
    p3 = geom.addPoint(L, H/2, 0, lc)
    p4 = geom.addPoint(0, H/2, 0, lc)

    l1 = geom.addLine(1, 2)
    l2 = geom.addLine(2, 3)
    l3 = geom.addLine(3, 4)
    l4 = geom.addLine(4, 1)

    rectangle = geom.addCurveLoop([1, 2, 3, 4])

    # add triangle
    #     6
    #    /|
    # 5 / |
    #   \ |
    #    \|
    #     7
    p5 = geom.addPoint(flame_loc-D*sqrt(3)/2,  0, 0, lc)
    p6 = geom.addPoint(flame_loc, +D/2, 0, lc)
    p7 = geom.addPoint(flame_loc, -D/2, 0, lc)

    l5 = geom.addLine(5, 7)
    l6 = geom.addLine(7, 6)
    l7 = geom.addLine(6, 5)

    # s1 = geom.addPlaneSurface([1])
    triangle = geom.addCurveLoop([5,6,7])

    geom.addPlaneSurface([rectangle, triangle])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [1], 2) # Bottom Wall 
    gmsh.model.addPhysicalGroup(1, [2], 3) # Outlet
    gmsh.model.addPhysicalGroup(1, [3], 4) # Upper Wall
    gmsh.model.addPhysicalGroup(1, [4], 1) # Inlet
    gmsh.model.addPhysicalGroup(1, [5,6,7], 5) # Triangle Walls



    gmsh.model.addPhysicalGroup(2, [1], 0)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.SaveAll", 0)
    # gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    gmsh.write("{}.msh".format(file))

    if fltk:
        fltk_options()
        gmsh.fltk.run()

    gmsh.finalize()


def fltk_options():

    # Type of entity label (0: description,
    #                       1: elementary entity tag,
    #                       2: physical group tag)
    gmsh.option.setNumber("Geometry.LabelType", 2)

    gmsh.option.setNumber("Geometry.PointNumbers", 0)
    gmsh.option.setNumber("Geometry.LineNumbers", 1)
    gmsh.option.setNumber("Geometry.SurfaceNumbers", 1)
    gmsh.option.setNumber("Geometry.VolumeNumbers", 0)

    # Mesh coloring(0: by element type, 1: by elementary entity,
    #                                   2: by physical group,
    #                                   3: by mesh partition)
    gmsh.option.setNumber("Mesh.ColorCarousel", 2)

    gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)

    gmsh.option.setNumber("Mesh.VolumeEdges", 0)
    gmsh.option.setNumber("Mesh.VolumeFaces", 0)


if __name__ == '__main__':

    geometry(fltk=True)
