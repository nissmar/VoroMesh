# C++ utilites for VoroMesh

This folder contains two utilities:
- `voromesh`: compute the VoroMesh from a list of labeled generators (*.vmesh file). Re-label infinite cells as outside. 
- `self_intersect`: print the number of self-intersections of a given triangle mesh.

## Compilation instructions

You need [CGAL](https://www.cgal.org) to compile

From ```src/cpp_utils```, copy and paste the following instructions in your terminal: 
```
mkdir build
cd build
cmake ../
make
```


To extract a triangle mesh from a list of generators, run: 

```
./voromesh ../bunny64.vmesh ../bunny64.off
```

To compute the number of self-intersections, run:

```
./self_intersect ../bunny64.off
```

The console should display ```0```