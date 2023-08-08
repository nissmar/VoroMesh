# VoroMesh

PyTorch implementation of the ICCV2023 paper **VoroMesh: Learning Watertight Surface Meshes with Voronoi Diagrams** 
[Nissim Maruani](https://nissmar.github.io), [Roman Klokov](https://scholar.google.ru/citations?user=LzkFOcoAAAAJ&hl=ru), [Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/), [Pierre Alliez](https://team.inria.fr/titane/pierre-alliez/), [Mathieu Desbrun](https://pages.saclay.inria.fr/mathieu.desbrun/).

### [Project Page](https://nissmar.github.io/voromesh.github.io/)

<img src='banner.png' />

## Citation

If you find our work useful in your research, please consider citing:

	@InProceedings{maruani23iccv,
	  author    = {N. Maruani and R. Klokov and M. Ovsjanikov and P. Alliez and M. Desbrun},
	  title     = {VoroMesh: Learning Watertight Surface Meshes with Voronoi Diagrams},
	  booktitle = {ICCV},
	  year      = {2023},
	}

## Requirements

The code is tested on the listed versions but other versions may also work:

- Python 3.9
- [PyTorch 1.12.1](https://pytorch.org/get-started/locally/)
- [PyTorch3D 0.7.0](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- [MinkowskiEngine 0.5.4](https://github.com/NVIDIA/MinkowskiEngine#anaconda)
- [trimesh 3.15.5](https://trimsh.org/install.html)
- tqdm 4.64.1
- yaml 0.2.5
- [CGAL](https://www.cgal.org) (Optional, for fast and accurate VoroMesh extraction)


## Datasets and pre-trained weights

Both ABC and Thingi32 preprocessed datasets are available [here](https://drive.google.com/file/d/1KO4Bhbz7yAlSx_p35rktdWGc7r69uLpw/view?usp=drive_link), as well as pretrained model checkpoints [here](https://drive.google.com/file/d/1Rw06bM-t88azkwoq6NtTVtJ4h7TVpdLs/view?usp=drive_link). Both should be unpacked in the root directory of the project for further use.

## Direct Optimization

python src/batch_voromesh.py

python src/eval.py


## Learning-based reconstruction from SDF input

 Training for different input SDF resolutions could be started with:
 ```
./scripts/train_voromesh_32.sh
./scripts/train_voromesh_64.sh
./scripts/train_voromesh_32+64.sh
```

Shape prediction for ABC with different resolution models:
```
./evaluate_voromesh_32_32_ABC.sh
./evaluate_voromesh_64_64_ABC.sh
./evaluate_voromesh_32+64_32_ABC.sh
./evaluate_voromesh_32+64_64_ABC.sh
```

Shape prediction for Thingi32 with different resolution models:
```
./evaluate_voromesh_32_32_thingi32.sh
./evaluate_voromesh_32_64_thingi32.sh
./evaluate_voromesh_32_128_thingi32.sh
./evaluate_voromesh_32+64_32_thingi32.sh
./evaluate_voromesh_32+64_64_thingi32.sh
./evaluate_voromesh_32+64_128_thingi32.sh
```

## Watertight test

TODO

