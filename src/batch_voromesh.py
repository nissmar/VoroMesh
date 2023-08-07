from time import time
from create_single_voromesh import create_voromesh, DEFAULTS
import os
from tqdm import tqdm
import torch
import utils.mesh_tools as mt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


src_dir = '../../data/thingy32/groundtruths/'  # where Thingi32 is stored
grid_n = 32
out_dir = "../out/batch_voronoi_{}_NDC_norm/".format(grid_n)
T0 = time()
opt_time = 0

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

for model_name in tqdm(os.listdir(src_dir)):
    v, f, samples = mt.load_and_sample_shape(
        model_name, src_dir, DEFAULTS["sample_fac"]*grid_n**2, rescale_f='NDC')
    V, t0 = create_voromesh(samples, v, f, grid_n)
    opt_time += t0

    # export .pt pytorch file
    # torch.save(
    #     V, 'voronois/batch_voronoi_{}_NDC_norm/voronoi_{}.pt'.format(grid_n, model_name[:-4]))

    # export mesh
    mt.export_vmesh(V.points.cpu().detach().numpy(),
                    V.values.cpu().detach().numpy(), 'temp')
    os.command(
        'cpp_utils/build/voromesh temp.vmesh {}/{}.off'.format(out_dir, model_name[:-4]))
print('TOTAL TIME: {}, OPT_TIME: {}'.format(time()-T0, opt_time))
