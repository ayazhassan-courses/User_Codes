# Distributed GPU Computing

 A typical computing cluster supports distributed GPU workloads on both single and multiple nodes, making it well-suited for deep learning, large language model (LLM) training, and large-scale GPU-accelerated simulations. On a single node, jobs can scale efficiently across multiple GPUs using MPI+CUDA or NCCL+CUDA, providing high-throughput parallelism and optimized GPU-to-GPU communication. For multi-node execution, the same frameworks extend naturally over high-speed interconnect, allowing seamless scaling from a few GPUs to dozens of nodes.

Common workflows include multi-GPU training with MPI and NCCL-based collectives for intra- and inter-node communication. Large-scale LLM training typically combines NCCL with MPI for process orchestration, ensuring efficient gradient synchronization across GPUs. Example job scripts in this repository show how to request GPUs with Slurm, configure CUDA/NCCL/MPI environments, and launch distributed jobs for both single-node and multi-node runs, serving as templates for adapting to specific workloads.

## Examples

* [Using MPI and CUDA](./MPI_and_CUDA/)
* [Using NCCL and CUDA on a single node](./NCCL_and_CUDA/SingleNode)
* [Using NCCL and CUDA on multi-node](./NCCL_and_CUDA/MultiNode)
* [Compiling and running KHARMA (MPI+CUDA code)](https://github.com/fasrc/User_Codes/tree/master/Applications/KHARMA)

# REQUIRED CHANGES TO RUN THE User_Codes REPO ON THE KFUPM JRCAI SLURM CLUSTER
## (GPU_Computing / Dist_GPU_Computing)

## CONTENTS
  0. General notes (apply to all sections)
  1. MPI and CUDA
  2. Distributed GPU Collectives on Multiple Nodes (NCCL + CUDA)
  3. Multi-GPU Computing with NCCL - Single Node
  4. PyTorch folder


### -------------------------------------------------------------------------------
### 0. GENERAL NOTES (APPLY TO ALL SECTIONS)
### -------------------------------------------------------------------------------

0.1 Partition name

  Before submitting any .sbatch / .slurm script, edit it and change the
  partition from "gpu" to "A5000":

      #SBATCH -p gpu              ->   #SBATCH -p A5000
      #SBATCH --partition=gpu     ->   #SBATCH --partition=A5000

  Reason: according to the KFUPM cluster documentation, jrcai12 belongs to
  the A5000 partition and has four GPUs.

  How to edit a file with nano:
      nano <file>
      (make the change)
      ctrl + o   then   enter     (save)
      ctrl + x                     (exit)

0.2 NCCL version mismatch (one-time install)

  nvcc on the cluster is CUDA 12.2, while the system NCCL needs CUDA 12.3 or
  newer. Linking with the system NCCL (-lnccl) fails. Install a newer NCCL
  once:

      pip install --user nvidia-nccl-cu12

0.3 NCCL_DIR variable

  NCCL_DIR is set only for the current terminal. Redo it every time the
  terminal is closed / reopened:

      NCCL_DIR=$(python -c "import nvidia.nccl as m; print(m.__path__[0])")

  To check it:
      echo $NCCL_DIR
      ls $NCCL_DIR/include $NCCL_DIR/lib


### -------------------------------------------------------------------------------
### 1. MPI AND CUDA
### -------------------------------------------------------------------------------

1.1 Makefile

  Problem: the Makefile uses MPI_INCLUDE and MPI_LIB, but neither is
  defined, which produces empty -I and -L flags.

  Fix:
      nano Makefile

  Add these lines before INCLUDES:

      MPI_INCLUDE ?= /opt/anaconda3/include
      MPI_LIB     ?= /opt/anaconda3/lib

  The new Makefile should therefore be:

      # Compiler
      NVCC = nvcc

      # Output and source
      TARGET = mpi_cuda.x
      SRC = mpi_cuda.cu

      # Compiler flags
      CXXFLAGS = -Xcompiler -fopenmp

      # MPI paths
      MPI_INCLUDE ?= /opt/anaconda3/include
      MPI_LIB     ?= /opt/anaconda3/lib

      INCLUDES = -I$(MPI_INCLUDE)
      LIBS = -L$(MPI_LIB) -Xlinker -rpath -Xlinker $(MPI_LIB) -lmpi -lgomp

      all: $(TARGET)

      $(TARGET): $(SRC)
      	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDES) $(LIBS)

      clean:
      	rm -f $(TARGET)

      .PHONY: all clean

  Save (ctrl + o, enter), exit (ctrl + x), then build:

      make

1.2 Running run_4.sbatch
      nano run_4.sbatch
      change:  #SBATCH -p gpu
      to:      #SBATCH -p A5000

  Then submit:
      sbatch run_4.sbatch


### -------------------------------------------------------------------------------
### 2. DISTRIBUTED GPU COLLECTIVES ON MULTIPLE NODES (NCCL + CUDA)
### -------------------------------------------------------------------------------

2.1 Compiling ncclAllGather_mpi.cu

  Original command (in the docs):
      nvcc -O3 -std=c++17 -o ncclAllGather_mpi.x ncclAllGather_mpi.cu -lnccl -lmpi

  Step 1 - Adding the MPI paths resolves only the paths issue:
      nvcc -O3 -std=c++17 \
        ncclAllGather_mpi.cu -o ncclAllGather_mpi.x \
        -I/opt/anaconda3/include \
        -L/opt/anaconda3/lib \
        -Xlinker -rpath -Xlinker /opt/anaconda3/lib \
        -lnccl -lmpi

  Step 2 - Another issue then appears: CUDA version mismatch (nvcc is CUDA
  12.2, and the system NCCL needs 12.3 or newer). Install the newer NCCL
  and set NCCL_DIR (see 0.2 and 0.3, ignore if you already did):
      pip install --user nvidia-nccl-cu12
      NCCL_DIR=$(python -c "import nvidia.nccl as m; print(m.__path__[0])")
      echo $NCCL_DIR
      ls $NCCL_DIR/include $NCCL_DIR/lib

  Step 3 - Correct compile command:
      nvcc -O3 -std=c++17 ncclAllGather_mpi.cu -o ncclAllGather_mpi.x \
        -I$NCCL_DIR/include -I/opt/anaconda3/include \
        -L$NCCL_DIR/lib -L/opt/anaconda3/lib \
        -Xlinker -rpath -Xlinker $NCCL_DIR/lib -Xlinker -rpath -Xlinker /opt/anaconda3/lib \
        -l:libnccl.so.2 -lmpi

  Step 4 - Verify:
      ldd ./ncclAllGather_mpi.x | grep -E 'nccl|cudart'

  The ldd line should show libnccl.so.2 coming from the ~/.local/... path.
  It should not list libcudart at all.

2.2 Running run_allgather_mpi.sbatch
      nano run_allgather_mpi.sbatch
      change:  #SBATCH -p gpu
      to:      #SBATCH -p A5000

  Then submit:
      sbatch run_allgather_mpi.sbatch

2.3 Compiling ncclAllGather_nompi.cu (and other files with the same error)
  If you face the previous error and already installed the newer version:

      NCCL_DIR=$(python -c "import nvidia.nccl as m; print(m.__path__[0])")

      nvcc -O3 -std=c++17 ncclAllGather_nompi.cu -o ncclAllGather_nompi.x \
        -I$NCCL_DIR/include -L$NCCL_DIR/lib \
        -Xlinker -rpath -Xlinker $NCCL_DIR/lib \
        -l:libnccl.so.2

  Remember to edit the .sbatch file to change -p gpu to -p A5000.


### -------------------------------------------------------------------------------
### 3. MULTI-GPU COMPUTING WITH NCCL - SINGLE NODE
### -------------------------------------------------------------------------------

  Assuming the pip install was already done (see 0.2):

      NCCL_DIR=$(python -c "import nvidia.nccl as m; print(m.__path__[0])")
      nvcc -O3 -std=c++17 ncclAllGather.cu -o ncclAllGather.x \
        -I$NCCL_DIR/include -L$NCCL_DIR/lib \
        -Xlinker -rpath -Xlinker $NCCL_DIR/lib \
        -l:libnccl.so.2

  For the other files, follow only the second instruction (the nvcc
  command, with the file name changed). NCCL_DIR is set only once per
  terminal and must be redone if the terminal is closed.

  Don't forget:
      nano run_ncclAllGather.slurm
      replace --partition=gpu with --partition=A5000


### -------------------------------------------------------------------------------
### 4. PYTORCH FOLDER
### -------------------------------------------------------------------------------

4.1 Install PyTorch
      pip install torch

4.2 mlp_ddp.py - create an sbatch file
      touch run_mlp_ddp.sbatch
      nano run_mlp_ddp.sbatch

  Add this content:

      #!/bin/bash
      #SBATCH --job-name=mlp_ddp
      #SBATCH --nodes=1
      #SBATCH --ntasks-per-node=2        # one process per GPU
      #SBATCH --gres=gpu:2
      #SBATCH --cpus-per-task=4          # becomes num_workers in the DataLoader
      #SBATCH --time=00:10:00
      #SBATCH --output=mlp_ddp_%j.out

      export PATH=/opt/anaconda3/bin:$PATH
      export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
      export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
      export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
      export NCCL_DEBUG=WARN

      srun python mlp_ddp.py

  Then submit:
      sbatch run_mlp_ddp.sbatch

4.3 random.py
  Works fine. Just run:
      python random.py

===============================================================================
