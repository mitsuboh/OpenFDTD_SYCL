/*
sample2.c

OpenFDTDデータ作成ライブラリ、サンプルプログラム No.2

コンパイル+実行:
Windows + VC++:
> cl.exe /O2 sample2.c ofd_datalib.c
> sample2.exe
Linux + gcc:
$ gcc -O sample2.c ofd_datalib.c -o sample2
$ ./sample2
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ofd_datalib.h"

int main(void)
{
	double x1, x2;
	double y1, y2, y3;
	double z0, z1, z2, z3, z4, z5;
	char   str[BUFSIZ], cmd[BUFSIZ];
	const double d = 5e-3;		// cell size
	const int    oc = 5;		// outer cells
	const char name[] = "sample2";

	// delete output files

	sprintf(str, "%s.log", name);
	remove(str);

#ifdef _WIN32
	sprintf(str, "%s.ev2", name);
	remove(str);
#endif

	// loop

	for (int loop = 1; loop <= 5; loop++) {

		// initialize

		ofd_init();

		// title

		sprintf(str, "%s_%03d", name, loop);
		ofd_title(str);

		// mesh

		x1 = -(3 + loop) * d;
		x2 = 0e-3;
		ofd_xsection(4, x1 - (oc * d), x1, x2, x2 + (oc * d));
		ofd_xdivision(3, oc, NINT(x2 - x1, d), oc);

		y1 = -50e-3;
		y2 = 0e-3;
		y3 = +50e-3;
		ofd_ysection(5, y1 - (oc * d), y1, y2, y3, y3 + (oc * d));
		ofd_ydivision(4, oc, NINT(y2 - y1, d), NINT(y3 - y2, d), oc);

		z0 = -75e-3;
		z1 = -50e-3;
		z2 = -25e-3;
		z3 = +25e-3;
		z4 = +50e-3;
		z5 = +75e-3;
		ofd_zsection(6, z0, z1, z2, z3, z4, z5);
		ofd_zdivision(5, 5, 5, 11, 5, 5);

		// geometry

		ofd_geometry(1, 1, x2, x2, y2, y2, z2, z3);

		ofd_geometry(1, 1, x1, x1, y1, y3, z1, z4);

		// feed

		ofd_feed('Z', x2, y2, 0e-3, 1, 0, 50);

		// frequency

		ofd_frequency1(2e9, 3e9, 10);
		ofd_frequency2(3e9, 3e9, 0);

		// solver

		ofd_solver(1000, 100, 1e-3);

		// far1d field

		ofd_plotfar1d('Z', 72, 0);

		// output

		sprintf(str, "%s_%03d.ofd", name, loop);
		ofd_outdata(str);

		// solver, post, append result

#ifdef _WIN32
		sprintf(cmd, "ofd.exe -n 4 %s", str);
		system(cmd);

		sprintf(cmd, "ofd_post.exe -n 4 %s", str);
		system(cmd);

		sprintf(cmd, "type ofd.log >> %s.log", name);
		system(cmd);

		sprintf(cmd, "type ev.ev2 >> %s.ev2", name);
		system(cmd);
#else
		sprintf(cmd, "./ofd -n 4 %s", str);
		system(cmd);

		sprintf(cmd, "./ofd_post -n 4 -html %s", str);
		system(cmd);

		sprintf(cmd, "cat ofd.log >> %s.log", name);
		system(cmd);

		sprintf(str, "%s_%03d.htm", name, loop);
		remove(str);
		rename("ev2d.htm", str);
#endif
	}

	return 0;
}
