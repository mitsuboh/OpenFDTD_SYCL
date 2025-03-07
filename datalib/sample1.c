/*
sample1.c

OpenFDTDデータ作成ライブラリ、サンプルプログラム No.1

コンパイル+実行:
Windows + VC++:
> cl.exe /O2 sample1.c ofd_datalib.c
> sample1.exe
Linux + gcc:
$ gcc -O sample1.c ofd_datalib.c -o sample1
$ ./sample1
*/

#include "ofd_datalib.h"

int main(void)
{
	// initialize

	ofd_init();

	// title

	ofd_title("sample1");

	// mesh

	ofd_xsection(2, -75e-3, +75e-3);
	ofd_xdivision(1, 30);

	ofd_ysection(2, -75e-3, +75e-3);
	ofd_ydivision(1, 30);

	ofd_zsection(4, -75e-3, -25e-3, +25e-3, +75e-3);
	ofd_zdivision(3, 10, 11, 10);

	// material

	ofd_material(2.0, 0.0, 1.0, 0.0, "");

	// geometry

	ofd_geometry(1, 1, 0e-3, 0e-3, 0e-3, 0e-3, -25e-3, +25e-3);

	// feed

	ofd_feed('Z', 0e-3, 0e-3, 0e-3, 1, 0, 50);
	//ofd_rfeed(10);

	// ABC

	//ofd_pml(5, 2, 1e-5);

	// frequency

	ofd_frequency1(2e9, 3e9, 10);
	ofd_frequency2(3e9, 3e9, 0);

	// solver

	ofd_solver(1000, 100, 1e-3);

	// iteration

	ofd_plotiter(1);

	// waveform and spectrum

	//ofd_plotfeed(1);
	//ofd_plotpoint(1);

	// frequency

	ofd_plotsmith(1);
	ofd_plotzin(1, 0, 0, 0);
	ofd_plotyin(1, 0, 0, 0);
	ofd_plotref(1, 0, 0, 0);
	//ofd_plotfar0d(90, 0, 1, 0, 0, 0);

	// far-1d

	ofd_plotfar1d('X', 72, 0);

	// far-2d

	ofd_plotfar2d(18, 36);

	// near-1d

	ofd_plotnear1d("E", 'Z', 30e-3, 0e-3);

	// near-2d

	ofd_plotnear2d("E", 'X', 30e-3);

	// output options

	//ofd_far1dstyle(1);
	//ofd_far1dcomponent(1, 0, 0);
	//ofd_far1ddb(1);
	//ofd_far1dnorm();
	//ofd_far1dscale(-30, +10, 4);
	//ofd_far2dcomponent(1, 0, 0, 0, 0, 0, 0);
	//ofd_far2ddb(1);
	//ofd_far2dscale(-30, +10);
	//ofd_far2dobj(0.5);
	//ofd_near1ddb(1);
	//ofd_near1dscale(-30, +10, 4);
	//ofd_near2ddim(1, 1);
	//ofd_near2ddb(1);
	//ofd_near2dscale(-30, +10);
	//ofd_near2dcontour(0);
	//ofd_near2dobj(1);
	//ofd_near2dzoom(-50e-3, 50e-3, -50e-3, 50e-3);
	//ofd_window2d(750, 500, 15, 0);
	//ofd_window3d(600, 600, 12, 60, 30);

	// output

	ofd_outdata("sample1.ofd");

	return 0;
}
