import dataclasses

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf
from isaaclab.sim.utils import bind_physics_material
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.stage import get_current_stage

from badmintonlab_game.assets.court import BadmintonCourtCfg
from badmintonlab_game.assets.robots.booster import BOOSTER_T1_CFG, T1_ACTION_SCALE

from . import mdp
from .player0_cfg import EventsCfg as Player0EventsCfg
from .player0_cfg import ObsCfg as Player0ObsCfg
from .player1_cfg import EventsCfg as Player1EventsCfg
from .player1_cfg import ObsCfg as Player1ObsCfg

_COURT_CFG = BadmintonCourtCfg()


def spawn_urdf_with_racket_material(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    prim = spawn_from_urdf(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    material_name = prim.GetPath().pathString.rsplit("/", 1)[-1].lower()
    material_path = f"/World/PhysicsMaterials/{material_name}_racket"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.2,
        dynamic_friction=0.15,
        restitution=0.75,
        friction_combine_mode="average",
        restitution_combine_mode="max",
    )
    material_cfg.func(material_path, material_cfg)
    racket_prim = get_current_stage().GetPrimAtPath(f"{prim.GetPath().pathString}/racket_link")
    pending = [racket_prim]
    while pending:
        child = pending.pop()
        if child.IsInstance():
            child.SetInstanceable(False)
        pending.extend(child.GetChildren())
    bind_physics_material(
        f"{prim.GetPath().pathString}/racket_link",
        material_path,
        stage=get_current_stage(),
    )
    return prim


def _racket_sensor(side: int) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot{side}/racket_link",
        history_length=4,
        track_air_time=False,
        force_threshold=1.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Shuttle"],
    )


@configclass
class DualAgentSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot_0: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot0")
    robot_1: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    shuttle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shuttle",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), roughness=0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.05,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00519),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.4,
                restitution=0.05,
                friction_combine_mode="average",
                restitution_combine_mode="min",
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 0.0, 1.55)),
    )
    racket_contact_0 = _racket_sensor(0)
    racket_contact_1 = _racket_sensor(1)
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    def __post_init__(self):
        super().__post_init__()
        for robot in (self.robot_0, self.robot_1):
            robot.spawn = dataclasses.replace(
                robot.spawn, merge_fixed_joints=False, func=spawn_urdf_with_racket_material
            )


@configclass
class DualActionsCfg:
    player_0 = mdp.JointPositionActionCfg(
        asset_name="robot_0", joint_names=[".*"], use_default_offset=True, scale=T1_ACTION_SCALE
    )
    player_1 = mdp.JointPositionActionCfg(
        asset_name="robot_1", joint_names=[".*"], use_default_offset=True, scale=T1_ACTION_SCALE
    )


@configclass
class DualObservationsCfg:
    player_0: Player0ObsCfg = Player0ObsCfg()
    player_1: Player1ObsCfg = Player1ObsCfg()


@configclass
class DualEventsCfg(Player0EventsCfg, Player1EventsCfg):
    reset_shuttle_pose = EventTerm(
        func=mdp.reset_shuttle_drop_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("shuttle"),
            "canonical_position": (-0.5 * _COURT_CFG.half_length + 0.8, 0.0, 2.2),
        },
    )


@configclass
class DualAgentBadmintonEnvCfg(ManagerBasedEnvCfg):
    scene: DualAgentSceneCfg = DualAgentSceneCfg(num_envs=2048, env_spacing=15.0)
    observations: DualObservationsCfg = DualObservationsCfg()
    actions: DualActionsCfg = DualActionsCfg()
    events: DualEventsCfg = DualEventsCfg()
    shuttle_mass = 0.00519
    shuttle_drag_k = 0.00110
    gravity_z = -9.81
    shuttle_landing_detection_height = 0.03
    hit_contact_threshold = 1.0
    max_serve_height = 1.15

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.court = _COURT_CFG
        from badmintonlab_game.assets.court import create_badminton_court

        for asset_name, asset_cfg in create_badminton_court(self.court).items():
            setattr(self.scene, asset_name, asset_cfg)
