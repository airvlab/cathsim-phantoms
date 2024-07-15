import os
import shutil
from pathlib import Path

import mujoco
import trimesh
import yaml
from lxml import etree
from mujoco import viewer
from trimesh.decomposition import convex_decomposition

VHACD_EXECUTABLE = shutil.which("testVHACD")

if VHACD_EXECUTABLE is None:

    def install_vhacd():
        response = input("ONLY PROCEED IF THE SIYSTEM IS LINUX\nProceed? (N/y):")
        if response != "y":
            exit()
        cmd = """
        set -xe;
        rm -f testVHACD;
        wget https://github.com/mikedh/v-hacd-1/raw/master/bin/linux/testVHACD;
        echo "e1e79b2c1b274a39950ffc48807ecb0c81a2192e7d0993c686da90bd33985130  testVHACD" | sha256sum --check;
        chmod +x testVHACD;
        sudo mv testVHACD /usr/bin/;
        """
        os.system(cmd)

    print("VHACD executable not found. Please install VHACD.")
    install_vhacd()


def load_config(path: Path) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Convert STL to MJCF")
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="meshes/",
        required=True,
        help="Folder with STL files",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=1_000_000,
        help="Maximum number of voxels generated during the voxelization stage(range=10, 000-16, 000, 000)",
    )
    parser.add_argument(
        "-mh",
        "--maxhulls",
        type=int,
        help="Maximum number of convex hulls to produce",
        default=100_000,
    )
    parser.add_argument(
        "--concavity",
        type=float,
        help="Maximum allowed concavity(range=0.0-1.0)",
        default=0.0025,
    )
    parser.add_argument(
        "--regex",
        type=str,
        default=".stl",
        help="Select which STL to decompose based on REGEX",
    )
    parser.add_argument(
        "-pd",
        "--planeDownsampling",
        type=int,
        default=4,
        help='Controls the granularity of the search for the "best" clipping plane(range=1-16)',
    )
    parser.add_argument(
        "-cd",
        "--convexhullDownsampling",
        type=int,
        default=4,
        help="Controls the precision of the convex-hull generation process during the clipping plane selection stage(range=1-16)",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.05,
        help="Controls the bias toward clipping along symmetry planes(range=0.0-1.0)",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=0.05,
        help="Controls the bias toward clipping along revolution axes(range=0.0-1.0)",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.00125,
        help="Controls the maximum allowed concavity during the merge stage(range=0.0-1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Controls the bias toward maximaxing local concavity(range=0.0-1.0)",
    )
    parser.add_argument(
        "-mnv",
        "--maxNumVerticesPerCH",
        type=int,
        default=64,
        help="Controls the maximum number of triangles per convex-hull(range=4-1024)",
    )
    parser.add_argument(
        "-mv",
        "--minVolumePerCH",
        type=float,
        default=0.0001,
        help="Controls the adaptive sampling of the generated convex-hulls(range=0.0-0.01)",
    )
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        default=False,
        help="Testing if MuJoCo compilation is successfull",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Allow V-HACD output?",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output/",
        help="Output folder for the convex hulls and xml file",
    )
    parser.add_argument(
        "-d",
        "--decompose",
        action="store_true",
        default=False,
        help="Use convex decomposition?",
    )
    parser.add_argument(
        "-vm",
        "--visualize",
        action="store_true",
        default=False,
        help="Visualize the model in MuJoCo. Note: requires the visualize flag as well",
    )
    return parser.parse_args()


def create_mjcf(
    collision_hulls: list, output_folder_path: str, mesh_name: str = None
) -> etree.ElementTree:
    """create mjcf file from collision hulls"""

    def validate_meshes(collision_hulls: list, treshold: float = 1e-6) -> list:
        for hull in collision_hulls:
            if hull.volume < treshold:
                collision_hulls.remove(hull)
        return collision_hulls

    root = etree.Element("mujoco")
    root.set("model", mesh_name)
    compiler = etree.SubElement(root, "compiler")
    compiler.set("meshdir", "meshes")

    default = etree.SubElement(root, "default")
    default_geom = etree.SubElement(default, "geom")
    default_geom.set("type", "mesh")

    asset = etree.SubElement(root, "asset")
    worldbody = etree.SubElement(root, "worldbody")
    body = etree.SubElement(worldbody, "body")
    body.set("name", "phantom")

    mesh_asset = etree.Element("mesh")
    mesh_asset.set("name", "visual")
    mesh_asset.set("file", f"{mesh_name}/visual.stl")
    asset.append(mesh_asset)

    geom = etree.Element("geom")
    geom.set("name", "visual")
    geom.set("mesh", "visual")
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    body.append(geom)

    for i in range(len(collision_hulls)):
        convex_hull_name = f"hull_{i}"
        convex_hull_mesh = etree.Element("mesh")
        convex_hull_mesh.set("name", convex_hull_name)
        convex_hull_mesh.set("file", f"{mesh_name}/{convex_hull_name}.stl")
        asset.append(convex_hull_mesh)

        convex_hull_geom = etree.Element("geom")
        convex_hull_geom.set("mesh", convex_hull_name)
        body.append(convex_hull_geom)

    tree = etree.ElementTree(root)
    etree.indent(tree, space="  ", level=0)

    return tree


def test_compile(mesh_name: str):
    try:
        model = mujoco.MjModel.from_xml_path(mesh_name + ".xml")
        data = mujoco.MjData(model)
        while data.time < 1:
            mujoco.mj_step(model, data)
    except Exception as e:
        print(e)
        print("Error compiling model.")


def export_meshes(
    collision_hulls: list, output_folder_path: Path, mesh_name: str = None
) -> None:
    """export hulls to stl files"""
    for i, hull in enumerate(collision_hulls):
        export_name = f"hull_{i}.stl"
        hull.export(output_folder_path / export_name)


def validate_meshes(
    collision_hulls: list, output_folder_path: Path, mesh_name: str = None
) -> trimesh.Trimesh:
    """validate meshes using trimesh"""
    while True:
        try:
            model = mujoco.MjModel.from_xml_path(
                output_folder_path / f"{mesh_name}.xml"
            )
            data = mujoco.MjData(model)
            while data.time < 1:
                mujoco.mj_step(model, data)
        except Exception as e:
            # get the name of the hull that caused the error
            hull_name = e.args[0].split(" ")[-1]
            hull_index = int(hull_name.split("_")[-1].split(".")[0])
            # remove the hull from the list
            collision_hulls.pop(hull_index)

    return collision_hulls


def main():
    args = parse_args()
    dir_path = Path(args.folder)
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not dir_path.exists():
        print(f"Folder {dir_path} does not exist.")
        exit(1)

    for file in dir_path.glob("*.stl"):
        mesh_name = file.stem
        output_folder_path = output_folder / "meshes" / mesh_name

        print(f"\n...Processing {file}...")

        if args.decompose:
            if output_folder_path.exists() and output_folder_path.iterdir():
                cont = input(
                    f"Output directory {output_folder_path.as_posix()} already exists and is not empty. Do you want to overwrite it? [y/n] "
                )
                if cont != "y":
                    exit(0)

            shutil.rmtree(output_folder_path, ignore_errors=True)
            mesh = trimesh.load_mesh(file)
            if len(mesh.faces) > 2e5:
                print(
                    f"Mesh {mesh_name} has more than 200k faces ({len(mesh.faces)}). Skipping."
                )
                continue

            output_folder_path.mkdir(parents=True, exist_ok=True)

            print(f"\n...Decomposing {mesh_name}...")

            convex_hulls = convex_decomposition(
                mesh=mesh,
                debug=args.verbose,
                resolution=args.resolution,
                maxhulls=args.maxhulls,
                concavity=args.concavity,
                planeDownsampling=args.planeDownsampling,
                convexhullDownsampling=args.convexhullDownsampling,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                delta=args.delta,
                maxNumVerticesPerCH=args.maxNumVerticesPerCH,
                minVolumePerCH=args.minVolumePerCH,
            )

            export_meshes(convex_hulls, output_folder_path, mesh_name)

            mesh.export(output_folder_path / "visual.stl")

            print(f"\n...Creating MJCF for {mesh_name}...")

            tree = create_mjcf(convex_hulls, output_folder_path, mesh_name)

            with open(output_folder / (mesh_name + ".xml"), "wb") as files:
                tree.write(files)

        if args.compile:
            print(f"\n...Compiling {mesh_name}...")
            test_compile((output_folder / mesh_name).as_posix())

            if args.visualize:
                print(f"\n...Visualizing {mesh_name}...")
                model = mujoco.MjModel.from_xml_path(
                    (output_folder / mesh_name).as_posix() + ".xml"
                )
                viewer.launch(model)

        print(f"\n...Done processing {file}...\n")


if __name__ == "__main__":
    main()
    path = Path(__file__).parent / "config.yaml"
    config = load_config(path.as_posix())
