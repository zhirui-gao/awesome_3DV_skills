import blenderproc as bproc
import argparse
import json
import os
import numpy as np

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import cv2
import os
import imageio
from typing import List
import bpy
from blenderproc.python.material import MaterialLoaderUtility
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.utility.Utility import Utility, resolve_path



class CCMaterialLoader:

    @staticmethod
    def create_material(new_mat: bpy.types.Material, base_image_path: str, ambient_occlusion_image_path: str,
                        metallic_image_path: str, roughness_image_path: str, alpha_image_path: str,
                        normal_image_path: str, displacement_image_path: str):
        """
        Create a material for the cctexture datatset, the combination used here is calibrated to this.

        :param new_mat: The new material, which will get all the given textures
        :param base_image_path: The path to the color image
        :param ambient_occlusion_image_path: The path to the ambient occlusion image
        :param metallic_image_path: The path to the metallic image
        :param roughness_image_path: The path to the roughness image
        :param alpha_image_path: The path to the alpha image (when this was written there was no alpha image provided \
                                 in the haven dataset)
        :param normal_image_path: The path to the normal image
        :param displacement_image_path: The path to the displacement image
        """
        nodes = new_mat.node_tree.nodes
        links = new_mat.node_tree.links

        principled_bsdf = Utility.get_the_one_node_with_type(nodes, "BsdfPrincipled")
        output_node = Utility.get_the_one_node_with_type(nodes, "OutputMaterial")

        collection_of_texture_nodes = []
        base_color = MaterialLoaderUtility.add_base_color(nodes, links, base_image_path, principled_bsdf)
        collection_of_texture_nodes.append(base_color)

        principled_bsdf.inputs["Specular"].default_value = 0.333

        ao_node = MaterialLoaderUtility.add_ambient_occlusion(nodes, links, ambient_occlusion_image_path,
                                                              principled_bsdf, base_color)
        collection_of_texture_nodes.append(ao_node)

        metallic_node = MaterialLoaderUtility.add_metal(nodes, links, metallic_image_path,
                                                        principled_bsdf)
        collection_of_texture_nodes.append(metallic_node)

        roughness_node = MaterialLoaderUtility.add_roughness(nodes, links, roughness_image_path,
                                                             principled_bsdf)
        collection_of_texture_nodes.append(roughness_node)

        alpha_node = MaterialLoaderUtility.add_alpha(nodes, links, alpha_image_path, principled_bsdf)
        collection_of_texture_nodes.append(alpha_node)

        normal_node = MaterialLoaderUtility.add_normal(nodes, links, normal_image_path, principled_bsdf,
                                                       invert_y_channel=True)
        collection_of_texture_nodes.append(normal_node)

        displacement_node = MaterialLoaderUtility.add_displacement(nodes, links, displacement_image_path,
                                                                   output_node)
        collection_of_texture_nodes.append(displacement_node)

        collection_of_texture_nodes = [node for node in collection_of_texture_nodes if node is not None]

        MaterialLoaderUtility.connect_uv_maps(nodes, links, collection_of_texture_nodes)



def load_ccmaterials(folder_path: str = "resources/cctextures", used_assets: list = None, preload: bool = False,
                     fill_used_empty_materials: bool = False, add_custom_properties: dict = None,
                     use_all_materials: bool = False) -> List[Material]:
    """ This method loads all textures obtained from https://cc0textures.com, use the script
    (scripts/download_cc_textures.py) to download all the textures to your pc.

    All textures here support Physically based rendering (PBR), which makes the textures more realistic.

    All materials will have the custom property "is_cc_texture": True, which will make the selection later on easier.

    :param folder_path: The path to the downloaded cc0textures.
    :param used_assets: A list of all asset names, you want to use. The asset-name must not be typed in completely, only the
                        beginning the name starts with. By default all assets will be loaded, specified by an empty list.
    :param preload: If set true, only the material names are loaded and not the complete material.
    :param fill_used_empty_materials: If set true, the preloaded materials, which are used are now loaded completely.
    :param add_custom_properties:  A dictionary of materials and the respective properties.
    :param use_all_materials: If this is false only a selection of probably useful textures is used. This excludes \
                              some see through texture and non tileable texture.
    :return a list of all loaded materials, if preload is active these materials do not contain any textures yet
            and have to be filled before rendering (by calling this function again, no need to save the prior
            returned list)
    """
    folder_path = resolve_path(folder_path)
    # this selected textures are probably useful for random selection
    probably_useful_texture = ["paving stones", "tiles", "wood", "fabric", "bricks", "metal", "wood floor",
                               "ground", "rock", "concrete", "leather", "planks", "rocks", "gravel",
                               "asphalt", "painted metal", "painted plaster", "marble", "carpet",
                               "plastic", "roofing tiles", "bark", "metal plates", "wood siding",
                               "terrazzo", "plaster", "paint", "corrugated steel", "painted wood", "lava"
                                                                                                   "cardboard", "clay",
                               "diamond plate", "ice", "moss", "pipe", "candy",
                               "chipboard", "rope", "sponge", "tactile paving", "paper", "cork",
                               "wood chips"]
    if not use_all_materials and used_assets is None:
        used_assets = probably_useful_texture
    elif used_assets is not None:
        used_assets = [asset.lower() for asset in used_assets]

    if add_custom_properties is None:
        add_custom_properties = dict()

    if preload and fill_used_empty_materials:
        raise Exception("Preload and fill used empty materials can not be done at the same time, check config!")

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        materials = []
        for asset in os.listdir(folder_path):
            if used_assets:
                skip_this_one = True
                for used_asset in used_assets:
                    # lower is necessary here, as all used assets are made that that way
                    if asset.lower().startswith(used_asset.replace(" ", "")):
                        skip_this_one = False
                        break
                if skip_this_one:
                    continue
            current_path = os.path.join(folder_path, asset)
            if os.path.isdir(current_path):
                base_image_path = os.path.join(current_path, "{}_2K-JPG_Color.jpg".format(asset))
                if not os.path.exists(base_image_path):
                    continue

                # construct all image paths
                ambient_occlusion_image_path = base_image_path.replace("Color", "AmbientOcclusion")
                metallic_image_path = base_image_path.replace("Color", "Metalness")
                roughness_image_path = base_image_path.replace("Color", "Roughness")
                alpha_image_path = base_image_path.replace("Color", "Opacity")
                normal_image_path = base_image_path.replace("Color", "Normal")
                displacement_image_path = base_image_path.replace("Color", "Displacement")

                # if the material was already created it only has to be searched
                if fill_used_empty_materials:
                    new_mat = MaterialLoaderUtility.find_cc_material_by_name(asset, add_custom_properties)
                else:
                    new_mat = MaterialLoaderUtility.create_new_cc_material(asset, add_custom_properties)

                # if preload then the material is only created but not filled
                if preload:
                    # Set alpha to 0 if the material has an alpha texture, so it can be detected e.q. in the material getter.
                    nodes = new_mat.node_tree.nodes
                    principled_bsdf = Utility.get_the_one_node_with_type(nodes, "BsdfPrincipled")
                    principled_bsdf.inputs["Alpha"].default_value = 0 if os.path.exists(alpha_image_path) else 1
                    # add it here for the preload case
                    materials.append(Material(new_mat))
                    continue
                elif fill_used_empty_materials and not MaterialLoaderUtility.is_material_used(new_mat):
                    # now only the materials, which have been used should be filled
                    continue

                # create material based on these image paths
                CCMaterialLoader.create_material(new_mat, base_image_path, ambient_occlusion_image_path,
                                                 metallic_image_path, roughness_image_path, alpha_image_path,
                                                 normal_image_path, displacement_image_path)

                materials.append(Material(new_mat))
        return materials
    else:
        raise Exception("The folder path does not exist: {}".format(folder_path))


def add_point_light(energe=3000, location=[-5, -5, 5]):
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(energe)
    # light.set_type("SUN")
    light.set_location(location)


def Fibonacci_grid_sample(num, radius):
    # https://www.jianshu.com/p/8ffa122d2c15
    points = [[0, 0, 0] for _ in range(num)]
    phi = 0.618
    for n in range(num):
        z = (2 * n - 1) / num - 1
        x = np.sqrt(np.abs(1 - z * z)) * np.cos(2 * np.pi * n * phi)
        y = np.sqrt(np.abs(1 - z * z)) * np.sin(2 * np.pi * n * phi)
        points[n][0] = x * radius
        points[n][1] = y * radius
        points[n][2] = z * radius

    points = np.array(points)
    return points


def sphere_angle_sample(num, radius):
    points = []
    for azim in np.linspace(-180, 180, num):
        elev = 60
        razim = np.pi * azim / 180
        relev = np.pi * elev / 180

        center = [0, 0, 0]
        xp = center[0] + np.cos(razim) * np.cos(relev) * radius
        yp = center[1] + np.sin(razim) * np.cos(relev) * radius
        zp = center[2] + np.sin(relev) * radius
        points.append([xp, yp, zp])
    points = np.array(points)
    return points

def sphere_angle_sample_for_video(num, radius):
    points = []
    for indice in np.linspace(0, 360, num):
        azim = indice
        elev = indice / 2
        if 90 < elev <= 180:
            elev = 180 - elev

        razim = np.pi * azim / 180
        relev = np.pi * elev / 180
        center = [0, 0, 0]
        xp = center[0] + np.cos(razim) * np.cos(relev) * radius
        yp = center[1] + np.sin(razim) * np.cos(relev) * radius
        zp = center[2] + np.sin(relev) * radius
        points.append([xp, yp, zp])
    points = np.array(points)
    return points



parser = argparse.ArgumentParser()

parser.add_argument('scene', help="Path to the scene.obj file, should be examples/resources/scene.obj")
parser.add_argument('output_dir', help="Path to where the final files, will be saved, could be examples/basics/basic/output")
parser.add_argument('num', default=100, type=int, help="number of rendering")
parser.add_argument('split', default="train", type=str, help="train, val or test")

args = parser.parse_args()
scene_name = os.path.basename(args.scene)[:-4]

bproc.init()

objs = bproc.loader.load_obj(args.scene)
obj = objs[0]
# Scale the 3D model
bb = obj.get_bound_box()
min_point, max_point = bb[0], None
max_dist = -1
for point in bb:
    dist = np.linalg.norm(point - min_point)
    if dist > max_dist:
        max_point = point
        max_dist = dist
diag = max_point - min_point

max_size = max(abs(diag[0]), abs(diag[1]), abs(diag[2]))  
print('diag:', diag)
print('max_size:', max_size)

scale = 1 / max_size
print("normalize scale:", scale)
obj.set_scale([scale, scale, scale])
poi = bproc.object.compute_poi(objs)
print("poi after scale:", poi)

obj.set_rotation_euler([0, 0, 0])
poi = bproc.object.compute_poi(objs)
print("poi after rotation:", poi)



# set_location = [0.5, 0.5, 0.5] - poi

# print("set_location:", set_location)
# obj.set_location(set_location)

# poi = bproc.object.compute_poi(objs)
# print("poi after set location:", poi)



# define the camera resolution
bproc.camera.set_resolution(800, 800)
angle_x,angle_y=bproc.camera.get_fov()   # just to get angle_x. see python/camera/CameraUtility.py for details or changes




# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(objs):
    obj.set_shading_mode('auto')
    obj.set_cp("instance_id", 1) 
    # print(obj.get_materials())
    # mat = obj.get_materials()[0]
    # if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
    #     grey_col = np.random.uniform(0.3, 0.9)   
    #     mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
    # mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    # mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[6, 6, 1]),
               bproc.object.create_primitive('PLANE', scale=[6, 6, 1], location=[0, -6, 6], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[6, 6, 1], location=[0, 6, 6], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[6, 6, 1], location=[6, 0, 6], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[6, 6, 1], location=[-6, 0, 6], rotation=[0, 1.570796, 0])]

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                   emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)
light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                        elevation_min = 5, elevation_max = 89, uniform_volume = False)
light_point.set_location(location)

# sample CC Texture and assign to room planes
cc_textures = load_ccmaterials('blenderproc/texture_dir')
random_cc_texture = np.random.choice(cc_textures)
for plane in room_planes:
    plane.replace_materials(random_cc_texture)

# Define a function that samples the initial pose of a given object above the ground
def sample_initial_pose(obj: bproc.types.MeshObject):
    obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))

# Sample objects on the given surface
placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=objs,
                                         surface=room_planes[0],
                                         sample_pose_func=sample_initial_pose,
                                         min_distance=0.01,
                                         max_distance=0.2)

# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)

poses = 0
while poses < args.num:
    # Sample location
    location = bproc.sampler.shell(center = [0, 0, 0],
                            radius_min = 2.5,
                            radius_max = 3.5,
                            elevation_min = 5,
                            elevation_max = 89,
                            uniform_volume = False)
    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    # poi = bproc.object.compute_poi(np.random.choice(placed_objects, size=1))
    poi = placed_objects[0].get_location() 
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    
    # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
        # Persist camera pose
        bproc.camera.add_camera_pose(cam2world_matrix)
        poses += 1




bproc.renderer.set_max_amount_of_samples(50)
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_normals_output()

bproc.renderer.enable_segmentation_output(
    map_by=["category", "instance", "name"],  # 按类别或实例生成Mask
    default_values={"category": 0}    # 地面等背景设为0
)
# # render the whole pipeline
data = bproc.renderer.render()

# save images

output_split_dir = os.path.join(args.output_dir, args.split)
os.makedirs(output_split_dir, exist_ok=True)
# import pdb; pdb.set_trace()
for i in range(bproc.utility.num_frames()):
    # rgb
    rgb = data["colors"][i]
    imageio.imwrite(f"{output_split_dir}/rgb_{i:03d}.png", rgb)  

    # 深度（EXR）
    depth_m = data["depth"][i]
    imageio.imwrite(f"{output_split_dir}/depth_{i:03d}.exr", depth_m.astype(np.float32))

    # 法线（EXR）——单位向量 [-1,1]，保留 float32 精度
    normal = data["normals"][i]                # 已经在 [-1,1]
    imageio.imwrite(f"{output_split_dir}/normal_{i:03d}.exr", normal.astype(np.float32))

    # 物体 mask —— segmentation["instance_segmaps"] 是实例 ID，["class_segmaps"] 是类别 ID
    class_map = data["instance_segmaps"][i].astype("uint16")
    foreground_mask = (class_map == 1).astype(np.uint8) * 255
    imageio.imwrite(f"{output_split_dir}/foreground_mask_{i:03d}.png", foreground_mask)
    



# Collect state of the camera at all frames
cam_states = []
for frame in range(bproc.utility.num_frames()):
    cam_states.append({
        "cam2world": bproc.camera.get_camera_pose(frame),
        "cam_K": bproc.camera.get_intrinsics_as_K_matrix()
    })
# Adds states to the data dict
data["cam_states"] = cam_states

# write the data to a .hdf5 container

# bproc.writer.write_hdf5(output_split_dir, data)

#存储相机参数到json文件
camera_json = {"camera_angle_x": angle_x, "location": placed_objects[0].get_location().tolist(), "frames": []}
for i in range(len(cam_states)):
    filename = f"rgb_{i:03d}"
    Rt = cam_states[i]["cam2world"]
    R = Rt[:3, :3]
    T = Rt[:3, 3:4]
    world2cam = np.concatenate((np.concatenate((R.transpose(), -(R.transpose().dot(T))), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
    K = cam_states[i]["cam_K"]
    camera = {"file_path": "./" + args.split + "/" + filename,
                  "rotation": 0,   # not use
                  "camera_intrinsics": K.tolist(),
                  "transform_matrix": Rt.tolist()}

    camera_json["frames"].append(camera)

with open(os.path.join(args.output_dir, "transforms_" + args.split + ".json"), "w") as f:
    json.dump(camera_json, f, indent=4)

# blenderproc run examples/datasets/abc_dataset/main_mv.py  examples/datasets/abc_dataset/00000006/00000006_d4fe04f0f5f84b52bd4f10e4_trimesh_001.obj examples/datasets/abc_dataset/output 100 train
# blenderproc vis hdf5 examples/datasets/abc_dataset/output/video/*.hdf5 --save examples/datasets/abc_dataset/output_rgb/


