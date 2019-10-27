import argparse
from cloudvolume import CloudVolume, Storage
from cloudvolume.lib import mkdir, touch, Vec
# from pathos.multiprocessing import ProcessingPool as Pool
import h5py
import numpy as np
import os


def ingest(args):
    """
    Ingest an HDF file to a CloudVolume bucket
    """
    if args.local_hdf_path:
        hdf_file = h5py.File(args.local_hdf_path, "r")
    else:
        with Storage(args.cloud_src_path) as storage:
            hdf_file = h5py.File(storage.get_file(args.cloud_hdf_filename), "r")
    cur_hdf_group = hdf_file
    for group_name in args.hdf_keys_to_dataset:
        cur_hdf_group = cur_hdf_group[group_name]
    hdf_dataset = cur_hdf_group
    if args.zyx:
        dataset_shape = np.array(
            [hdf_dataset.shape[2], hdf_dataset.shape[1], hdf_dataset.shape[0]]
        )
    else:
        dataset_shape = np.array([*hdf_dataset.shape])
    if args.layer_type == "image":
        data_type = "uint8"
    else:
        data_type = "uint64"
    voxel_offset = args.voxel_offset
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type=args.layer_type,
        data_type=data_type,
        encoding="raw",
        resolution=args.resolution,
        voxel_offset=voxel_offset,
        chunk_size=args.chunk_size,
        volume_size=dataset_shape,
    )
    provenance = {
        "description": args.provenance_description,
        "owners": [args.owner]
    }
    vol = CloudVolume(args.dst_path, info=info, provenance=provenance)
    vol.commit_info()
    vol.commit_provenance()

    all_files = set()
    for x in np.arange(
        voxel_offset[0], voxel_offset[0] + dataset_shape[0], args.chunk_size[0]
    ):
        for y in np.arange(
            voxel_offset[1], voxel_offset[1] + dataset_shape[1], args.chunk_size[1]
        ):
            for z in np.arange(
                voxel_offset[2], voxel_offset[2] + dataset_shape[2], args.chunk_size[2]
            ):
                all_files.add(tuple((x, y, z)))

    progress_dir = mkdir("progress/")  # unlike os.mkdir doesn't crash on prexisting
    done_files = set()
    for done_file in os.listdir(progress_dir):
        done_files.add(tuple(done_file.split(",")))
    to_upload = all_files.difference(done_files)

    for chunk_start_tuple in to_upload:
        chunk_start = np.array(list(chunk_start_tuple))
        end_of_dataset = np.array(voxel_offset) + dataset_shape
        chunk_end = chunk_start + np.array(args.chunk_size)
        chunk_end = Vec(*chunk_end)
        chunk_end = Vec.clamp(chunk_end, Vec(0, 0, 0), end_of_dataset)
        chunk_hdf_start = chunk_start - voxel_offset
        chunk_hdf_end = chunk_end - voxel_offset
        if args.zyx:
            chunk = hdf_dataset[
                chunk_hdf_start[2] : chunk_hdf_end[2],
                chunk_hdf_start[1] : chunk_hdf_end[1],
                chunk_hdf_start[0] : chunk_hdf_end[0],
            ]
            chunk = chunk.T
        else:
            chunk = hdf_dataset[
                chunk_hdf_start[0] : chunk_hdf_end[0],
                chunk_hdf_start[1] : chunk_hdf_end[1],
                chunk_hdf_start[2] : chunk_hdf_end[2],
            ]
        print("Processing ", chunk_start_tuple)
        array = np.array(chunk, dtype=np.dtype(data_type), order="F")
        vol[
            chunk_start[0] : chunk_end[0],
            chunk_start[1] : chunk_end[1],
            chunk_start[2] : chunk_end[2],
        ] = array
        touch(os.path.join(progress_dir, str(chunk_start_tuple)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_hdf_path", type=str, help="Local path to hdf file", required=False
    )
    # Either supply a local hdf path, OR supply a cloud hdf path and
    # a cloud hdf filename
    parser.add_argument(
        "--cloud_src_path",
        type=str,
        help="CloudVolume path to the bucket containing the hdf file",
        required=False,
    )
    parser.add_argument(
        "--cloud_hdf_filename",
        type=str,
        help="HDF filename on the cloud",
        required=False,
    )
    parser.add_argument(
        "--hdf_keys_to_dataset",
        type=str,
        nargs="+",
        help="Sequence of keys to retrieve the dataset from the hdf file. E.g. if hdf_file['volumes']['raw'] contains the image data, then you should pass 'volumes raw' (without quotes) as the argument",
    )
    parser.add_argument(
        "--zyx",
        action="store_true",
        default=False,
        help="Pass this argument if the data is stored in zyx order.",
    )
    parser.add_argument(
        "--dst_path", type=str, help="CloudVolume path to store the segmentation"
    )
    parser.add_argument("--layer_type", type=str, help="image or segmentation")
    parser.add_argument(
        "--chunk_size",
        type=int,
        nargs=3,
        help="cloudvolume chunk size. int list of length 3 space delimited (e.g. 128 128 256)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=3,
        help="voxel dimensions (nm). int list of length 3 space delimited (e.g. 4 4 40)",
    )
    parser.add_argument(
        "--voxel_offset",
        type=int,
        nargs=3,
        default=np.array([0,0,0])
    )
    parser.add_argument("--provenance_description", type=str, required=False)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="int number of processes to use for parallel ops",
    )
    parser.add_argument(
        "--owner", type=str, help="email address for cloudvolume provenance"
    )
    args = parser.parse_args()
    if args.layer_type != "image" and args.layer_type != "segmentation":
        raise ValueError("Invalid layer type (must be image or segmentation)")
    if not args.local_hdf_path and not (
        args.cloud_src_path and args.cloud_hdf_filename
    ):
        raise ValueError(
            "Must specify either a local_hdf_path or both a cloud path and cloud filename"
        )
    ingest(args)

# Example
# python hdf5_ingest.py --local_hdf_path groundtruth.hdf 
# --hdf_keys_to_dataset volumes labels neuron_ids
# --dst_path gs://example/cloud/path/
# --layer_type segmentation --chunk_size 512 512 32 --resolution 16 16 40 --owner johndoe@gmail.com
