PROGRAM = ../ofd_post.exe

OMPOPT  = /openmp

OBJS    =           \
Main.obj            \
post_data.obj       \
readout.obj         \
post.obj            \
outputFar0d.obj     \
outputFar1d.obj     \
outputFar2d.obj     \
outputNear1d.obj    \
outputNear2d.obj    \
calcFar1d.obj       \
calcFar2d.obj       \
calcNear1d.obj      \
calcNear2d.obj      \
plot2dIter.obj      \
plot2dFeed.obj      \
plot2dPoint.obj     \
plot2dFreq.obj      \
plot2dFreq0.obj     \
plot2dFar0d0.obj    \
plot2dFar1d0.obj    \
plot3dFar2d.obj     \
plot3dFar2d0.obj    \
plot2dNear1d0.obj   \
plot2dNear2d0.obj   \
plot3dNear2d0.obj   \
planewave.obj       \
../sol/nearfield_c.obj     \
../sol/farfield.obj        \
../sol/outputCoupling.obj  \
../sol/utils.obj           \
ev2dlib.obj         \
ev3dlib.obj         \
ev2d.obj            \
../sol/ev3d.obj

CC      = cl.exe
CFLAGS  = /O2 /nologo /I../include /W4 /wd4996
#CFLAGS  = $(CFLAGS) /D_DOUBLE
#CFLAGS  = $(CFLAGS) /D_ID32
LDFLAGS = /O2 /nologo
LIBS    =

$(PROGRAM): $(OBJS)
	@echo "Loading $(PROGRAM) ... "
	@$(CC) $(LDFLAGS) /Fe$(PROGRAM) $(OBJS) $(LIBS)

Main.obj:
	$(CC) /c $(CFLAGS) $(OMPOPT) $?
calcFar1d.obj:
	$(CC) /c $(CFLAGS) $(OMPOPT) $?
calcFar2d.obj:
	$(CC) /c $(CFLAGS) $(OMPOPT) $?

.c.obj:
	$(CC) /c $(CFLAGS) $<

clean:
	del *.obj
