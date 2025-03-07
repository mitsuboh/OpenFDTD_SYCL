PROGRAM = ../ofd_mpi.exe

OMPOPT  = /openmp
MPIOPT  = /D_MPI

OBJS    =           \
Main.obj            \
solve.obj           \
setup_mpi.obj       \
comm.obj            \
comm_X.obj          \
comm_Y.obj          \
comm_Z.obj          \
../sol/updateEx.obj        \
../sol/updateEy.obj        \
../sol/updateEz.obj        \
../sol/updateHx.obj        \
../sol/updateHy.obj        \
../sol/updateHz.obj        \
../sol/dispersionEx.obj    \
../sol/dispersionEy.obj    \
../sol/dispersionEz.obj    \
../sol/input_data.obj      \
../sol/input2.obj          \
../sol/monitor.obj         \
../sol/memallocfree1.obj   \
../sol/memallocfree2.obj   \
../sol/memallocfree3.obj   \
../sol/setup.obj           \
../sol/setupSize.obj       \
../sol/setup_vector.obj    \
../sol/setupId.obj         \
../sol/setupDispersion.obj \
../sol/setupDft.obj        \
../sol/setupMurHx.obj      \
../sol/setupMurHy.obj      \
../sol/setupMurHz.obj      \
../sol/setupPmlEx.obj      \
../sol/setupPmlEy.obj      \
../sol/setupPmlEz.obj      \
../sol/setupPmlHx.obj      \
../sol/setupPmlHy.obj      \
../sol/setupPmlHz.obj      \
../sol/setupPml.obj        \
../sol/initfield.obj       \
../sol/vfeed.obj           \
../sol/efeed.obj           \
../sol/eload.obj           \
../sol/vpoint.obj          \
../sol/average.obj         \
../sol/murH.obj            \
../sol/pmlEx.obj           \
../sol/pmlEy.obj           \
../sol/pmlEz.obj           \
../sol/pmlHx.obj           \
../sol/pmlHy.obj           \
../sol/pmlHz.obj           \
../sol/pbcx.obj            \
../sol/pbcy.obj            \
../sol/pbcz.obj            \
../sol/dftNear3d.obj       \
../sol/nearfield_c.obj     \
../sol/outputChars.obj     \
../sol/outputZin.obj       \
../sol/outputSpara.obj     \
../sol/outputCoupling.obj  \
../sol/outputCross.obj     \
../sol/ingeometry.obj      \
../sol/geomlines.obj       \
../sol/farfield.obj        \
../sol/writeout.obj        \
../sol/utils.obj           \
../sol/plot3dGeom.obj      \
../sol/ev3d.obj

CC      = cl.exe
CFLAGS  = /O2 /nologo /I../include /W4 /wd4996
#CFLAGS  = $(CFLAGS) /D_DOUBLE
#CFLAGS  = $(CFLAGS) /D_ID32
LDFLAGS = /O2 /nologo
LIBS    = msmpi.lib

$(PROGRAM): $(OBJS)
	@echo "Loading $(PROGRAM) ... "
	@$(CC) $(LDFLAGS) /Fe$(PROGRAM) $(OBJS) $(LIBS)

Main.obj:
	$(CC) /c $(CFLAGS) $(OMPOPT) $?
comm.obj:
	$(CC) /c $(CFLAGS) $(MPIOPT) $?
comm_X.obj:
	$(CC) /c $(CFLAGS) $(MPIOPT) $?
comm_Y.obj:
	$(CC) /c $(CFLAGS) $(MPIOPT) $?
comm_Z.obj:
	$(CC) /c $(CFLAGS) $(MPIOPT) $?

.c.obj:
	$(CC) /c $(CFLAGS) $<

clean:
	del *.obj
