PROGRAM = ../ofd_cuda.exe

TRASH   = ..\ofd_cuda.exp ..\ofd_cuda.lib

OBJS    =             \
Main.obj              \
info_gpu.obj          \
check_gpu.obj         \
cuda_memory.obj       \
setup_gpu.obj         \
memallocfree2_gpu.obj \
memallocfree3_gpu.obj \
solve.obj             \
initfield.obj         \
updateEx.obj          \
updateEy.obj          \
updateEz.obj          \
updateHx.obj          \
updateHy.obj          \
updateHz.obj          \
dispersionEx.obj      \
dispersionEy.obj      \
dispersionEz.obj      \
efeed.obj             \
eload.obj             \
vpoint.obj            \
murH.obj              \
pmlEx.obj             \
pmlEy.obj             \
pmlEz.obj             \
pmlHx.obj             \
pmlHy.obj             \
pmlHz.obj             \
pbcx.obj              \
pbcy.obj              \
pbcz.obj              \
dftNear3d.obj         \
average.obj           \
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
../sol/cputime.obj           \
../sol/plot3dGeom.obj        \
../sol/ev3d.obj

CC      = nvcc.exe
CFLAGS  = -O2 -I../include
#CFLAGS  = $(CFLAGS) -D_DOUBLE
#CFLAGS  = $(CFLAGS) -D_ID32
LDFLAGS = -O2
LIBS    =

$(PROGRAM): $(OBJS)
	@echo "Loading $(PROGRAM) ... "
	@$(CC) -o $(PROGRAM) $(LDFLAGS) $(OBJS) $(LIBS)
	del $(TRASH)

.SUFFIXES: .cu

.cu.obj:
	$(CC) -c $(CFLAGS) $<

clean:
	del *.obj
