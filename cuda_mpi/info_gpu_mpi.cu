/*
info_gpu_mpi.cu (CUDA + MPI) (OpenFDTD/OpenTHFD/OpenSTF)

check GPU, set device, show info
*/

#include <stdio.h>
#include <string.h>

extern int rank2device(int, int, const int []);
extern int check_gpu(int, char []);

extern "C" {
extern void comm_check(int, int, int);
extern void comm_string(const char *, char *);
}

void info_gpu_mpi(FILE *fp, int nhost, const int ndevice[], int gpu, int um, int commsize, int commrank, int prompt)
{
	if (gpu) {
		char lstr[2 * BUFSIZ], msg[BUFSIZ];
		char *gstr = (char *)malloc(commsize * BUFSIZ * sizeof(char));

		const int device = rank2device(commrank, nhost, ndevice);

		const int ierr = check_gpu(device, msg);
		sprintf(lstr, "  GPU-%d: %s, U.M.%s, device=%d", commrank, msg, (um ? "ON" : "OFF"), device);
		comm_string(lstr, gstr);
		if (commrank == 0) {
			fprintf(fp,     "%s\n", gstr);
			fprintf(stdout, "%s\n", gstr);
			fflush(fp);
			fflush(stdout);
		}
		comm_check(ierr, 1, prompt);

		free(gstr);
	}
}
