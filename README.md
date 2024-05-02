## Design/Approach

Following are the key design points used for this implementation.

* The dialect primarily uses [LLVM](https://mlir.llvm.org/docs/Dialects/LLVM/) and [Func](https://mlir.llvm.org/docs/Dialects/Func/) dialects. And the operations are lowered on to the same.
* The Pandas/MLIR dialect in this project is implemented on top of an SQLite backend(implemented in ``pandas-mlir\lib\DataLib``).
  * The dataframe operations get lowered down to ``llvm.call`` and other related operations to execute the underlying SQLite operations to perform dataframe modifications.
  * The library internally allocates instance of DataFrame corresponding to the dataframes that are requested to be created.
  * The library is built using ``sqlite-amalgamation-3450200`` version of [SQLite3](https://sqlite.org/amalgamation.html) library.


Motivation for this approach is derived from other opensource project of [LLVMSQLite](https://github.com/KowalskiThomas/LLVMSQLite) listed below, but the implementation is done ground up and no source code has been reused in to this project 



### Advantages

* While implementing this as backend based approach, SQLLite can be replaced with other DataFrame or Database backend which are optimized for specific usecases or dataset. Requiring only the DataLib to be ported/modified.

* Makes used of matured (data optimized) implementation of SQLite or similar engines than relying on a ground up implementation.

### Limitations

* Making use of Dialect/constructs like Affine loops and Tensor will enable extensive compiler/MLIR level optimization prospects. While on the other hand, this approach will have limited prospects in that manner. 


### Sourcecode Layout

Code/pandas-mlir

* ``include`` : Dialect related headers
* ``lib``:
  * ``DataLib``: Backend library to support SQLLite based dataframe management. The main library APIs can be seen in ``lib\DataLib\src\DataFrame.c``.
  * ``PandasMLIR``: Dialect related functionality implementation
* ``pandasmlir-opt``: Implementation of Pandas-MLIR opt tool for applying passes to MLIR and lowering from one dialect to another.

### Alternative Approaches investigated

* An initial approach was made to make use of Tensors/Memrefs to allocate and manage series(columns) of the table, but was unsucessful due to bugs.

---
## Build Instructions

Inorder to build project source code, LLVM/MLIR toolchain need to be built. Instructions for the same are listed in next sub-section.

### LLVM/MLIR Build 

Clone llvm-project-release-17.x source code in to a directory.
* The build shall be done outside the source tree (e.g: ``build/mlir-build`` in the below example)
* The installation directory is also kept separate and provided along as ``-DCMAKE_INSTALL_PREFIX=../../opt`` in the build command listing below

```bash
$ pwd
/home/vaisakhps/developer/Compiler/E0_255-CD_MLIR-Project/build/mlir-build

$ cmake  \
	-DCMAKE_BUILD_TYPE=Release  \
	-DLLVM_TARGETS_TO_BUILD=host  \
	-DCMAKE_INSTALL_PREFIX=../../opt \
	-DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;cross-project-tests;lld;lldb;openmp;mlir" \
	-DLLVM_PARALLEL_COMPILE_JOBS=12 -DLLVM_PARALLEL_LINK_JOBS=5 \
	-DLLVM_INCLUDE_TESTS=ON -DLLVM_BUILD_TESTS=ON \
	-DLLVM_INCLUDE_TOOLS=ON -DLLVM_BUILD_TOOLS=ON \
	-DLLVM_INCLUDE_EXAMPLES=ON -DLLVM_BUILD_EXAMPLES=ON \
	-DLLVM_INCLUDE_BENCHMARKS=ON -DLLVM_BUILD_BENCHMARKS=ON \
	-DLLVM_BUILD_DOCS=ON  \
	../../llvm-project-release-17.x/llvm

$ cmake --build . --parallel 12 && cmake --install .
```

More information available at https://github.com/llvm/llvm-project/blob/main/llvm/docs/CMake.rst

### Project source build

The project is implemented as an out-of-tree MLIR Standalone project with ``examples/standalone`` as reference. The same can be built with cmake with two additional arguments. These arguements helps in search for MLIR and LLVM's SDK exports.

* ``MLIR_DIR`` containing path to ``/lib/cmake/mlir/`` that has the cmake files for build directory (``$BUILD_DIR_PREFIX`` is the full path on to that directory)
* Similarly, ``LLVM_DIR`` containing path to ``lib/cmake/llvm/``.

```bash
$ export BUILD_DIR_PREFIX=/home/vaisakhps/developer/Compiler/E0_255-CD_MLIR-Project/build/mlir-build/
$ pwd
/home/vaisakhps/developer/Compiler/E0_255-CD_MLIR-Project/pandas-mlir/build

$ mkdir build && cd build

$ cmake -DMLIR_DIR=$BUILD_DIR_PREFIX/lib/cmake/mlir/ -DLLVM_DIR=$BUILD_DIR_PREFIX/lib/cmake/llvm/  ..

$ cmake --build .
```

---
## Running test cases

Tests for JIT execution of input query can be done using ``pandasmlir-opt`` utility generated in build directory of project.

```bash
$ ./bin/pandasmlir-opt -emit=jit  ../test/llvm-trial.mlir
```
