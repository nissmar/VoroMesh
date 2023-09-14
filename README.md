# VoroMesh: Learning Watertight Surface Meshes with Voronoi Diagrams
[Nissim Maruani](https://nissmar.github.io), [Roman Klokov](https://scholar.google.ru/citations?user=LzkFOcoAAAAJ&hl=ru), [Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/), [Pierre Alliez](https://team.inria.fr/titane/pierre-alliez/), [Mathieu Desbrun](https://pages.saclay.inria.fr/mathieu.desbrun/).


PyTorch implementation of the ICCV2023 paper ([project page](https://nissmar.github.io/voromesh.github.io/))



<img src='banner.png' />

## Requirements

The code is tested on the listed versions but other versions may also work:

- Python 3.9
- [PyTorch 1.12.1](https://pytorch.org/get-started/locally/)
- [PyTorch3D 0.7.0](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- [MinkowskiEngine 0.5.4](https://github.com/NVIDIA/MinkowskiEngine#anaconda)
- [trimesh 3.15.5](https://trimsh.org/install.html)
- igl 2.2.1
- tqdm 4.64.1
- yaml 0.2.5
- [CGAL](https://www.cgal.org) (Optional, for fast and accurate VoroMesh extraction)

## Compiling C++ utilities (optional)

In theory, our VoroMesh is guaranteed to be watertight. In practice, however, discretization can potentialy create self-intersections. To alleviate this problem, we use [CGAL](https://www.cgal.org) EPIC (exact predicate inexact construction) kernel for both our VoroMesh construction and self-intersection check. In our tests, 100% of our meshes are watertight. These tools can be compiled in `src/cpp_utils`.  

If the C++ utilities are not compiled, the code will still run but the output shapes will probably **NOT BE WATERTIGHT**.

## Direct Optimization

Our optimized models can be downloaded [here](https://drive.google.com/file/d/1XbiEPSMEwGFW4zxkvoXf9Vz78cSzoPny/view?usp=drive_link).

To fit a VoroMesh to a single shape:

```
python src/single_voromesh.py ./data/bunny.stl
```

To process the Thingi32 dataset with resolutions 32, 64 and 128:

```
python src/batch_voromesh.py
```

To compute the relevant metrics:

```
python src/eval/eval_THINGI.py out/direct_voromesh_32/
python src/eval/eval_THINGI.py out/direct_voromesh_64/
python src/eval/eval_THINGI.py out/direct_voromesh_128/
```


## Datasets and pre-trained weights

Preprocessed datasets [ABC](https://drive.google.com/file/d/1YwF-iUfvKTRaR1ZNyQ_TX2IIN597LeGK/view?usp=drive_link) (10.4 Gb) and [Thingi32](https://drive.google.com/file/d/1V77k18e8yQpyzvmBIhtpZIXNfPpi6Zmg/view?usp=drive_link) (0.7 Gb) should be unpacked in the `./data` directory, pretrained model [checkpoints](https://drive.google.com/file/d/1Rw06bM-t88azkwoq6NtTVtJ4h7TVpdLs/view?usp=drive_link) (0.4 Gb) - in the `./` directory of the project.



## Learning-based reconstruction from SDF input


To evaluate our pretrained models on ABC with different resolutions:
```
./scripts/evaluate_voromesh_32_32_ABC.sh
./scripts/evaluate_voromesh_64_64_ABC.sh
./scripts/evaluate_voromesh_32+64_32_ABC.sh
./scripts/evaluate_voromesh_32+64_64_ABC.sh
```

To compute the relevant metrics:
```
python src/eval/eval_ABC.py samples/ABC_32_32
python src/eval/eval_ABC.py samples/ABC_64_64
python src/eval/eval_ABC.py samples/ABC_32+64_32
python src/eval/eval_ABC.py samples/ABC_32+64_64
```

To evaluate our pretrained models on Thingi32 with different resolutions:
```
./scripts/evaluate_voromesh_32_32_thingi32.sh
./scripts/evaluate_voromesh_32_64_thingi32.sh
./scripts/evaluate_voromesh_32_128_thingi32.sh
./scripts/evaluate_voromesh_32+64_32_thingi32.sh
./scripts/evaluate_voromesh_32+64_64_thingi32.sh
./scripts/evaluate_voromesh_32+64_128_thingi32.sh
```


To compute the relevant metrics:
```
python src/eval/eval_THINGI.py samples/Thingi32_32 -formatted 1
python src/eval/eval_THINGI.py samples/Thingi32_64 -formatted 1
python src/eval/eval_THINGI.py samples/Thingi32_128 -formatted 1
python src/eval/eval_THINGI.py samples/Thingi32_32+64_32 -formatted 1
python src/eval/eval_THINGI.py samples/Thingi32_32+64_64 -formatted 1
python src/eval/eval_THINGI.py samples/Thingi32_32+64_128 -formatted 1
```

 To train with different input SDF resolutions:
 ```
./scripts/train_voromesh_32.sh
./scripts/train_voromesh_64.sh
./scripts/train_voromesh_32+64.sh
```
## Watertight check (requires C++ utilities)

To check for watertightness:

```
python src/eval/check_watertight.py out/direct_voromesh_32
```


## Citation
If you find our work useful in your research, please consider citing:

	@InProceedings{maruani23iccv,
	  author    = {N. Maruani and R. Klokov and M. Ovsjanikov and P. Alliez and M. Desbrun},
	  title     = {VoroMesh: Learning Watertight Surface Meshes with Voronoi Diagrams},
	  booktitle = {ICCV},
	  year      = {2023},
	}