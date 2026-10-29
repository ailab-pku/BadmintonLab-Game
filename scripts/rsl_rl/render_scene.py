"""Render-only court dressing; no colliders or changes to match dynamics."""

from pathlib import Path

import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


class MatchVisuals:
    """Dress the existing court and track the physical shuttle with a feather mesh."""

    def __init__(self, env, shuttle_scale=1.5):
        self.env = env
        self.stage = omni.usd.get_context().get_stage()
        self.root = "/World/MatchVisuals"
        UsdGeom.Xform.Define(self.stage, self.root)

        # Visibility changes leave the original terrain/net/shuttle physics intact.
        for path in ("/World/ground", "/World/envs/env_0/Net", "/World/envs/env_0/Shuttle"):
            UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).MakeInvisible()

        ground = self._material("Ground", (0.055, 0.070, 0.085))
        court = self._material("Court", (0.035, 0.32, 0.17))
        net = self._material("Net", (0.025, 0.030, 0.035))
        tape = self._material("Tape", (0.92, 0.92, 0.87))
        self._box("Surround", (200.0, 200.0, 0.02), (0, 0, -0.009), ground)
        UsdShade.MaterialBindingAPI.Apply(self.stage.GetPrimAtPath("/World/ground")).Bind(
            ground, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
        UsdShade.MaterialBindingAPI.Apply(self.stage.GetPrimAtPath("/World/envs/env_0/Floor")).Bind(
            court, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

        # Open netting, with the same height and sag as the rule geometry.
        cfg = env.cfg.court
        half_width = cfg.doubles_half_width

        def top(y):
            return cfg.net_center_height + (cfg.net_edge_height - cfg.net_center_height) * abs(y) / half_width

        bottom = cfg.net_edge_height - cfg.net_visual_height
        ys = [-half_width + i * cfg.doubles_width / 84 for i in range(85)]
        strands = [[(cfg.net_x, y, bottom), (cfg.net_x, y, top(y))] for y in ys]
        strands += [[(cfg.net_x, y, bottom + (top(y) - bottom) * row / 11) for y in ys] for row in range(11)]
        self._curves("Netting", strands, 0.007, net)
        self._curves("TopTape", [[(cfg.net_x, y, top(y)) for y in ys]], 0.035, tape)
        self._curves("BottomCord", [[(cfg.net_x, -half_width, bottom), (cfg.net_x, half_width, bottom)]], 0.008, net)

        # Broad overhead light and neutral fill keep white robot parts readable.
        dome = UsdLux.DomeLight(self.stage.GetPrimAtPath("/World/DomeLight"))
        dome.GetIntensityAttr().Set(250)
        dome.GetColorAttr().Set(Gf.Vec3f(0.9, 0.94, 1.0))
        key = UsdLux.RectLight.Define(self.stage, self.root + "/KeyLight")
        key.CreateIntensityAttr(600)
        key.CreateWidthAttr(12)
        key.CreateHeightAttr(8)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
        UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0, -1, 9))
        sun = UsdLux.DistantLight.Define(self.stage, self.root + "/SunLight")
        sun.CreateIntensityAttr(1500)
        sun.CreateAngleAttr(3)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-25, -25, 0))

        shuttle = UsdGeom.Xform.Define(self.stage, self.root + "/Shuttle")
        self.shuttle_position = shuttle.AddTranslateOp()
        self.shuttle_rotation = shuttle.AddOrientOp()
        shuttle.AddScaleOp().Set(Gf.Vec3f(shuttle_scale))
        mesh = UsdGeom.Xform.Define(self.stage, self.root + "/Shuttle/Feathers")
        mesh.GetPrim().GetReferences().AddReference(
            str(Path(__file__).parent / "assets/shuttlecock/shuttlecock_visual.usd")
        )
        mesh.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0056, -0.0650, 0.0046))
        self.update()

    def _material(self, name, color):
        path = self.root + "/Materials/" + name
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _box(self, name, size, position, material):
        box = UsdGeom.Cube.Define(self.stage, self.root + "/" + name)
        box.CreateSizeAttr(1.0)
        box.AddTranslateOp().Set(Gf.Vec3d(*position))
        box.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(material)

    def _curves(self, name, strands, width, material):
        curves = UsdGeom.BasisCurves.Define(self.stage, self.root + "/" + name)
        curves.CreateTypeAttr("linear")
        curves.CreateWrapAttr("nonperiodic")
        curves.CreateCurveVertexCountsAttr([len(strand) for strand in strands])
        curves.CreatePointsAttr([point for strand in strands for point in strand])
        curves.CreateWidthsAttr([width])
        curves.SetWidthsInterpolation("constant")
        UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(material)

    def update(self):
        shuttle = self.env.scene["shuttle"].data
        position = shuttle.root_pos_w[0].tolist()
        velocity = Gf.Vec3d(*shuttle.root_lin_vel_w[0].tolist())
        direction = velocity.GetNormalized() if velocity.GetLength() > 0.1 else Gf.Vec3d(0, 0, -1)
        self.shuttle_position.Set(Gf.Vec3d(*position))
        self.shuttle_rotation.Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), direction).GetQuat()))
