/*
sample3.c

OpenFDTDデータ作成ライブラリ、サンプルプログラム No.3

コンパイル+実行:
Windows + VC++:
> cl.exe /O2 sample3.c ofd_datalib.c
> sample3.exe
Linux + gcc:
$ gcc -O sample3.c ofd_datalib.c -o sample3
$ ./sample3
*/

#include "ofd_datalib.h"

int main(void)
{
	// initialize

	ofd_init();

	// title

	ofd_title("sample3");

	// mesh

	ofd_xsection(2, -150e-3, +150e-3);
	ofd_xdivision(1, 30);

	ofd_ysection(2, -150e-3, +150e-3);
	ofd_ydivision(1, 30);

	ofd_zsection(2, -150e-3, +150e-3);
	ofd_zdivision(1, 30);

	// material

	ofd_material(2.0, 0.0, 1.0, 0.0, "");

	// geometry

	ofd_geometry(1, 1, -50e-3, +50e-3, -50e-3, +50e-3, -50e-3, +50e-3);

	// planewave

	ofd_planewave(90, 0, 1);

	// ABC

	//ofd_pml(5, 2, 1e-5);

	// frequency

	ofd_frequency1(3e9, 3e9, 0);
	ofd_frequency2(3e9, 3e9, 0);

	// solver

	ofd_solver(1000, 50, 1e-3);

	// iteration

	ofd_plotiter(1);

	// far-0d

	//ofd_plotfar0d(90, 0);

	// far-1d

	ofd_plotfar1d('Z', 72, 0);

	// far-2d

	ofd_plotfar2d(18, 36);

	// near-1d

	ofd_plotnear1d("E", 'X', 0e-3, 0e-3);

	// near-2d

	ofd_plotnear2d("E", 'Z', 0e-3);

	// output options

	//ofd_far1dstyle(1);
	//ofd_far1dcomponent(1, 0, 0);
	//ofd_far1ddb(1);
	//ofd_far1dscale(-30, +10, 4);
	//ofd_far2dcomponent(1, 0, 0, 0, 0, 0,0);
	//ofd_far2ddb(1);
	//ofd_far2dscale(-30, +10);
	//ofd_far2dobj(0.5);
	//ofd_near1ddb(1);
	//ofd_near1dscale(-30, +10, 4);
	//ofd_near1dnoinc();
	//ofd_near2ddim(1, 1);
	//ofd_near2ddb(1);
	//ofd_near2dscale(-30, +10);
	//ofd_near2dcontour(0);
	//ofd_near2dobj(1);
	//ofd_near2dnoinc();

	// output

	ofd_outdata("sample3.ofd");

	return 0;
}
