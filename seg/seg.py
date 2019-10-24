import argparse
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec
import numpy as np
import waterz
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from time import strftime

def vec3(s):
    try:
        z, y, x = map(int, s.split(','))
        return (z,y,x)
    except:
        raise argparse.ArgumentTypeError("Vec3 must be z,y,x")

def segment(args):
    """Run segmentation on contiguous block of affinities from CV

    Args:
        args: ArgParse object from main
    """
    bbox_start = Vec(*args.bbox_start)
    bbox_size = Vec(*args.bbox_size)
    chunk_size = Vec(*args.chunk_size)
    bbox = Bbox(bbox_start, bbox_start + bbox_size)
    src_cv = CloudVolume(args.src_path, fill_missing=True, 
			 parallel=args.parallel)
    info = CloudVolume.create_new_info(
    	num_channels = 1,
    	layer_type = 'segmentation', 
    	data_type = 'uint64', 
    	encoding = 'raw', 
    	resolution = src_cv.info['scales'][args.mip]['resolution'], 
    	voxel_offset = bbox_start,
    	chunk_size = chunk_size,
    	volume_size = bbox_size,
        mesh = 'mesh_mip_{}_err_{}'.format(args.mip, 
					   args.max_simplification_error)
    )
    dst_cv = CloudVolume(args.dst_path, info=info, parallel=args.parallel)
    dst_cv.provenance.description = 'ws+agg using waterz'
    dst_cv.provenance.processing.append({
        'method': {
            'task': 'watershed+agglomeration',
            'src_path': args.src_path,
            'dst_path': args.dst_path,
            'mip': args.mip,
            'shape': bbox_size.tolist(),
            'bounds': [
                dst_bbox.minpt.tolist(),
                dst_bbox.maxpt.tolist(),
                ],
            },
        'by': args.owner,
        'date': strftime('%Y-%m-%d%H:%M %Z'),
        })
    dst_cv.provenance.owners = [args.owner]
    dst_cv.commit_info()
    dst_cv.commit_provenance()
    if args.segment:
        print('Downloading affinities') 
        aff = src_cv[bbox.to_slices()]
        aff = np.transpose(aff, (3,0,1,2))
        aff = np.ascontiguousarray(aff, dtype=np.float32)
        thresholds = [args.threshold]
        print('Starting ws+agg')
        seg_gen = waterz.agglomerate(aff, thresholds)
        seg = next(seg_gen)
        print('Deleting affinities')
        del aff
        print('Uploading segmentation')
        dst_cv[bbox.to_slices()] = seg
    if args.mesh:
        print('Starting meshing')
        with LocalTaskQueue(parallel=args.parallel) as tq:
            tasks = tc.create_meshing_tasks(layer_path=args.dst_path, 
                      mip=args.mip, 
                      shape=args.chunk_size,
                      simplification=True,
                      max_simplification_error=args.max_simplification_error,
                      progress=True)
            tq.insert_all(tasks)
            tasks = tc.create_mesh_manifest_tasks(layer_path=args.dst_path,
                                                  magnitude=args.magnitude)
            tq.insert_all(tasks)
        print("Meshing complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str,
	help='CloudVolume path for affinities')
    parser.add_argument('--dst_path', type=str,
	help='CloudVolume path to store the segmentation')
    parser.add_argument('--bbox_start', type=vec3,
	help='bbox origin, int list, commas & no space, e.g. x,y,z')
    parser.add_argument('--bbox_size', type=vec3,
	help='bbox size, int list, commas & no space, e.g. x,y,z')
    parser.add_argument('--mip', type=int, default=0,
	help='int MIP level for affinities')
    parser.add_argument('--parallel', type=int, default=1,
	help='int number of processes to use for parallel ops')
    parser.add_argument('--threshold', type=float, default=0.7,
	help='float for agglomeration threshold')
    parser.add_argument('--chunk_size', type=vec3,
	help='cloudvolume chunk, int list, commas & no space, e.g. x,y,z')
    parser.add_argument('--owner', type=str, 
      help='email address for cloudvolume provenance')
    parser.add_argument('--segment', 
      help='run segmentation on affinities',
      action='store_true')
    parser.add_argument('--mesh', 
      help='mesh existing segmentation',
      action='store_true')
    parser.add_argument('--max_simplification_error', type=int, default=40,
	help='int for mesh simplification')
    parser.add_argument('--magnitude', type=int, default=4,
	help='int for magnitude used in igneous mesh manifest')
    args = parser.parse_args()
    segment(args) 
