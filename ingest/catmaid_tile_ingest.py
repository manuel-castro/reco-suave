import argparse
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
import io 
import requests
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec
from time import strftime

def ingest(args):
    """Ingest CATMAID tiles with same row,col index across z range 

    Args:
	args: ArgParse object from main
    """
    mip = args.mip 
    row = args.row 
    col = args.col 
    chunk_size = Vec(*args.chunk_size)
    bbox = Bbox(bbox_start, bbox_start + bbox_size)
    chunk_size = [1024, 1024, 1]
    x_start = row*chunk_size[0]
    x_stop = (row+1)*chunk_size[0]
    y_start = col*chunk_size[1]
    y_stop = (col+1)*chunk_size[1]
    z_start = args.z_start
    z_stop = args.z_stop
    info = CloudVolume.create_new_info(
    	num_channels = 1,
    	layer_type = 'image', 
    	data_type = 'uint8', 
    	encoding = 'raw', 
    	resolution = [args.resolution[0]*2**mip, 
    		      args.resolution[1]*2**mip, 
		      args.resolution[2]]
    	voxel_offset = [x_start,
                            y_start, 
                            z_start],
    	chunk_size = chunk_size,
    	volume_size = [chunk_size[0],
                           chunk_size[1],
                           z_stop - z_start] 
    )
    x_range = range(x_start, x_stop)
    y_range = range(y_start, y_stop)
    z_range = range(z_start, z_stop)
    
    url_base = args.url_base 
    ex_url = '{}/{}/{}/{}/{}.jpg'.format(url_base, mip, z_start, row, col)
    vol = CloudVolume(args.dst_path, info=info)
    vol.provenance.description = 'Cutout from CATMAID'
    vol.provenance.processing.append({
        'method': {
            'task': 'ingest',
            'src_path': url_base,
            'dst_path': args.dst_path,
    	'row': row,
    	'col': col,
    	'z_range': [z_start, z_stop],
    	'chunk_size': chunk_size.tolist()
            'mip': args.mip,
            },
        'by': args.owner,
        'date': strftime('%Y-%m-%d%H:%M %Z'),
        })
    vol.provenance.owners = [args.owner]
    vol.commit_info()
    vol.commit_provenance()
    
    def process(z):
      url = '{}/{}/{}/{}/{}.jpg'.format(args.url_base, args.mip, z, row, col)
      data = requests.get(url)
      if data.status_code == 200:
        img = Image.open(io.BytesIO(data._content))
        width, height = img.size
        array = np.array(list(img.getdata()), dtype=np.uint8, order='F')
        array = array.reshape((1, height, width)).T
        vol[:, :, z:z+1] = array 
    
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        executor.map(process, z_range)
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url_base', type=str)
    parser.add_argument('--dst_path', type=str,
	help='CloudVolume path to store the segmentation')
    parser.add_argument('--row', type=int, 
	help='row index of CATMAID to ingest')
    parser.add_argument('--col', type=int, 
	help='column index of CATMAID to ingest')
    parser.add_argument('--z_start', type=int, 
	help='z index to start of range')
    parser.add_argument('--z_stop', type=int, 
	help='z index to end of range')
    parser.add_argument('--mip', type=int, default=0,
	help='int MIP level for affinities')
    parser.add_argument('--parallel', type=int, default=1,
	help='int number of processes to use for parallel ops')
    parser.add_argument('--chunk_size', type=vec3,
	help='cloudvolume chunk, int list, commas & no space, e.g. x,y,z')
    parser.add_argument('--owner', type=str, 
      help='email address for cloudvolume provenance')
    parser.add_argument('--resolution', type=vec3, default=[4,4,40],
	help='voxel dimensions (nm), int list, commas & no space, e.g. x,y,z')
    args = parser.parse_args()
    segment(args) 
