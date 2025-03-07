PROGRAM = ../ofd_cuda_mpi.exe

TRASH   = ..\ofd_cuda_mpi.exp ..\ofd_cuda_mpi.lib

#MPIOPT  = -D_MPI -I../../../../progra~2/micros~1/MPI/Include
MPIOPT  = -D_MPI -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include"

OBJS    =             \
Main.obj              \
solve.obj             \
comm_cuda_X.obj       \
comm_cuda_Y.obj       \
comm_cuda_Z.obj       \
info_gpu_mpi.obj      \
rank2device.obj       \
../cuda/check_gpu.obj         \
../cuda/cuda_memory.obj       \
../cuda/setup_gpu.obj         \
../cuda/memallocfree2_gpu.obj \
../cuda/memallocfree3_gpu.obj \
../cuda/initfield.obj         \
../cuda/updateEx.obj          \
../cuda/updateEy.obj          \
../cuda/updateEz.obj          \
../cuda/updateHx.obj          \
../cuda/updateHy.obj          \
../cuda/updateHz.obj          \
../cuda/dispersionEx.obj      \
../cuda/dispersionEy.obj      \
../cuda/dispersionEz.obj      \
../cuda/murH.obj              \
../cuda/pmlEx.obj             \
../cuda/pmlEy.obj             \
../cuda/pmlEz.obj             \
../cuda/pmlHx.obj             \
../cuda/pmlHy.obj             \
../cuda/pmlHz.obj             \
../cuda/pbcx.obj              \
../cuda/pbcy.obj              \
../cuda/pbcz.obj              \
../cuda/efeed.obj             \
../cuda/eload.obj             \
../cuda/vpoint.obj            \
../cuda/dftNear3d.obj         \
../cuda/average.obj           \
../sol/input_data.obj        \
../sol/input2.obj            \
../sol/monitor.obj           \
../sol/memallocfree1.obj     \
../sol/memallocfree3.obj     \
../sol/setup.obj             \
../sol/setupSize.obj         \
../sol/setup_vector.obj      \
../sol/setupId.obj           \
../sol/setupDispersion.obj   \
../sol/setupDft.obj          \
../sol/setupMurHx.obj        \
../sol/setupMurHy.obj        \
../sol/setupMurHz.obj        \
../sol/setupPmlEx.obj        \
../sol/setupPmlEy.obj        \
../sol/setupPmlEz.obj        \
../sol/setupPmlHx.obj        \
../sol/setupPmlHy.obj        \
../sol/setupPmlHz.obj        \
../sol/setupPml.obj          \
../sol/vfeed.obj             \
../sol/nearfield_c.obj       \
../sol/outputChars.obj       \
../sol/outputZin.obj         \
../sol/outputSpara.obj       \
../sol/outputCoupling.obj    \
../sol/outputCross.obj       \
../sol/ingeometry.obj        \
../sol/geomlines.obj         \
../sol/farfield.obj          \
../sol/writeout.obj          \
../sol/utils.obj             \
../sol/plot3dGeom.obj        \
../sol/ev3d.obj              \
../mpi/setup_mpi.obj         \
../mpi/comm.obj

CC      = nvcc.exe
CFLAGS  = -O2 -I../include
#CFLAGS  = $(CFLAGS) -D_DOUBLE
#CFLAGS  = $(CFLAGS) -D_ID32
LDFLAGS = -O2
LIBS    = msmpi.lib

$(PROGRAM): $(OBJS)
	@echo "Loading $(PROGRAM) ... "
	@$(CC) -o $(PROGRAM) $(LDFLAGS) $(OBJS) $(LIBS)
	del $(TRASH)

.SUFFIXES: .cu

comm_cuda_X.obj:
	$(CC) -c $(CFLAGS) $(MPIOPT) $?
comm_cuda_Y.obj:
	$(CC) -c $(CFLAGS) $(MPIOPT) $?
comm_cuda_Z.obj:
	$(CC) -c $(CFLAGS) $(MPIOPT) $?

.cu.obj:
	$(CC) -c $(CFLAGS) $<

clean:
	del *.obj
