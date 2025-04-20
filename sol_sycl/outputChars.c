/*
outputChars.c

calculate and output to ofd.log
*/

#include "ofd.h"
#include "complex.h"
#include "ofd_prototype.h"


void outputChars(int ilog, FILE *fp, const char fn_feed[], const char fn_point[])
{
	// (0) setup far field
	alloc_farfield();
	setup_farfield();

	// (1) feed data
	if (NFeed && NFreq1) {
		calcZin();
		if (ilog) {
			outputZin(stdout);
			outputZin(fp);
		}
		outputFeed(fn_feed);
		calcPin();  // for post
	}

	// (2) point data
	if (NPoint && NFreq1) {
		calcSpara();
		if (ilog) {
			outputSpara(stdout);
			outputSpara(fp);
		}
		outputPoint(fn_point);
	}

	// (3) coupling
	if (NFeed && NPoint && NFreq1) {
		if (ilog) {
			outputCoupling(stdout);
			outputCoupling(fp);
		}
	}

	// (4) cross section
	if (IPlanewave && NFreq2) {
		if (ilog) {
			outputCross(stdout);
			outputCross(fp);
		}
	}
}
