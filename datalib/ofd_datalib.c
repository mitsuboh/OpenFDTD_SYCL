/*
ofd_datalib.c
OpenFDTD datalib
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

#define PROGRAM "OpenFDTD 4 2"

int _MAXSECTION;
int _MAXMATERIAL;
int _MAXGEOMETRY;
int _MAXFEED;
int _MAXLOAD;
int _MAXPOINT;
int _MAXFAR1D;
int _MAXNEAR1D;
int _MAXNEAR2D;

typedef struct {
	int    db;              // 0/1 : [V/m]/[dB]
	int    user;            // 0/1 : auto/user
	double min, max;        // min, max
	int    div;             // division
} scale_t;                  // scale

char      _title[BUFSIZ];
int       _xsections, _ysections, _zsections;
int       _xdivisions, _ydivisions, _zdivisions;
double   *_xsection, *_ysection, *_zsection;
int      *_xdivision, *_ydivision, *_zdivision;
int       _materials;
int      *_material_type;
double  (*_material)[4];
char    **_material_name;
int       _geometrys;
int      *_geometry_m, *_geometry_shape;
double  (*_geometry)[8];
char    **_geometry_name;
int       _feeds;
char     *_feed_dir;
double  (*_feed_par)[6];
int       _loads;
char     *_load_dir, *_load_type;
double  (*_load_par)[4];
int       _planewave;
double    _planewave_theta;
double    _planewave_phi;
int       _planewave_pol;
int       _points;
char     *_point_dir;
double  (*_point_pos)[3];
char      _point_p[BUFSIZ];
double    _rfeed;
double    _pulsewidth;
double    _timestep;
double    _f1start, _f1end;
int       _f1div;
double    _f2start, _f2end;
int       _f2div;
int       _maxiter, _nout;
double    _converg;
int       _abc;
int       _pml_l;
double    _pml_m, _pml_r0;
int       _pbcx, _pbcy, _pbcz;
int       _matchingloss;
int       _plotiter;
int       _plotfeed;
int       _plotpoint;
int       _plotsmith;
int       _plotzin;
int       _plotyin;
int       _plotref;
int       _plotspara;
int       _plotcoupling;
int       _plotfar0d;
scale_t   _zinscale;
scale_t   _yinscale;
scale_t   _refscale;
scale_t   _sparascale;
scale_t   _couplingscale;
scale_t   _far0dscale;
double    _far0d_angle[2];
int       _freqdiv;
int       _far1ds;
char     *_far1d_dir;
int      *_far1d_div;
double   *_far1d_angle;
int       _far1dstyle;
int       _far1dcomponent[3];
int       _far1dnorm;
scale_t   _far1dscale;
int       _far2ds;
int       _far2d_div[2];
int       _far2dcomponent[7];
scale_t   _far2dscale;
double    _far2dobj;
int       _near1ds;
char    (*_near1d_cmp)[3];
char     *_near1d_dir;
double  (*_near1d_pos)[2];
scale_t   _near1dscale;
int       _near1dnoinc;
int       _near2ds;
char    (*_near2d_cmp)[3];
char     *_near2d_dir;
double   *_near2d_pos;
int       _near2ddim[2];
int       _near2dframe;
scale_t   _near2dscale;
int       _near2dcontour;
int       _near2dobj;
int       _near2dnoinc;
int       _near2dizoom;
double    _near2dpzoom[4];


// initialize
void ofd_init(void)
{
	_MAXSECTION  = 1000;
	_MAXMATERIAL = 256;
	_MAXGEOMETRY = 100000;
	_MAXFEED     = 1000;
	_MAXLOAD     = 1000;
	_MAXPOINT    = 1000;
	_MAXFAR1D    = 1000;
	_MAXNEAR1D   = 1000;
	_MAXNEAR2D   = 1000;

	_xsections = _ysections = _zsections = 0;
	_xdivisions = _ydivisions = _zdivisions = 0;
	_materials = _geometrys = _feeds = _planewave = _loads = _points = 0;
	_far1ds = _far2ds = _near1ds = _near2ds = 0;

	strcpy(_title, "");
	_pulsewidth = 0;
	_timestep   = 0;
	_rfeed      = 0;
	_f1start    = 3e8;
	_f1end      = 3e8;
	_f1div      = 0;
	_f2start    = 3e8;
	_f2end      = 3e8;
	_f2div      = 0;
	_maxiter    = 0;
	_nout       = 0;
	_converg    = 0;
	_abc        = 0;
	_pml_l      = 0;
	_pml_m      = 0;
	_pml_r0     = 0;
	_pbcx       =
	_pbcy       =
	_pbcz       = 0;
	strcpy(_point_p, "");

	_matchingloss = 0;
	_plotiter     =
	_plotfeed     =
	_plotpoint    =
	_plotsmith    =
	_plotzin      =
	_plotyin      =
	_plotref      =
	_plotspara    =
	_plotcoupling =
	_plotfar0d    = 0;
	_freqdiv = 10;
	_far1dstyle = 0;
	_far1dcomponent[0] = 1; _far1dcomponent[1] = _far1dcomponent[2] = 0;
	_far1dnorm = 0;
	_far1dscale.user = 0;
	_far1dscale.db = 1;
	_far1dscale.min = _far1dscale.max = 0;
	_far1dscale.div = 1;
	_far2dcomponent[0] = 1; _far2dcomponent[1] = _far2dcomponent[2] = _far2dcomponent[3] = _far2dcomponent[4] = _far2dcomponent[5] = _far2dcomponent[6] = 0;
	_far2dscale.user = 0;
	_far2dscale.db = 1;
	_far2dscale.min = _far2dscale.max = 0;
	_far2dobj = 0.5;
	_near1dscale.user = 0;
	_near1dscale.db = 0;
	_near1dscale.min = _near1dscale.max = 0;
	_near1dscale.div = 10;
	_near1dnoinc = 0;
	_near2ddim[0] = _near2ddim[1] = 1;
	_near2dframe = 0;
	_near2dscale.user = 0;
	_near2dscale.db = 0;
	_near2dscale.min = _near2dscale.max = 0;
	_near2dcontour = 0;
	_near2dobj = 1;
	_near2dnoinc = 0;
	_near2dizoom = 0;

	_xsection       = (double *)      malloc((_MAXSECTION + 1) * sizeof(double));
	_ysection       = (double *)      malloc((_MAXSECTION + 1) * sizeof(double));
	_zsection       = (double *)      malloc((_MAXSECTION + 1) * sizeof(double));
	_xdivision      = (int *)         malloc( _MAXSECTION      * sizeof(int));
	_ydivision      = (int *)         malloc( _MAXSECTION      * sizeof(int));
	_zdivision      = (int *)         malloc( _MAXSECTION      * sizeof(int));

	_material       = (double (*)[4]) malloc(_MAXMATERIAL * 4  * sizeof(double));
	_material_type  = (int *)         malloc(_MAXMATERIAL      * sizeof(int));
	_material_name  = (char **)       malloc(_MAXMATERIAL      * sizeof(char *));

	_geometry_m     = (int *)         malloc(_MAXGEOMETRY      * sizeof(int));
	_geometry_shape = (int *)         malloc(_MAXGEOMETRY      * sizeof(int));
	_geometry_name  = (char **)       malloc(_MAXGEOMETRY      * sizeof(char *));
	_geometry       = (double (*)[8]) malloc(_MAXGEOMETRY * 8 * sizeof(double));

	_feed_dir       = (char *)        malloc(_MAXFEED          * sizeof(char));
	_feed_par       = (double (*)[6]) malloc(_MAXFEED * 6      * sizeof(double));

	_load_dir       = (char *)        malloc(_MAXLOAD          * sizeof(char));
	_load_type      = (char *)        malloc(_MAXLOAD          * sizeof(char));
	_load_par       = (double (*)[4]) malloc(_MAXLOAD * 4      * sizeof(double));

	_point_dir      = (char *)        malloc(_MAXPOINT         * sizeof(char));
	_point_pos      = (double (*)[3]) malloc(_MAXPOINT * 3     * sizeof(double));

	_far1d_dir      = (char *)        malloc(_MAXFAR1D         * sizeof(char));
	_far1d_div      = (int *)         malloc(_MAXFAR1D         * sizeof(int));
	_far1d_angle    = (double *)      malloc(_MAXFAR1D         * sizeof(double));

	_near1d_cmp     = (char (*)[3])   malloc(_MAXNEAR1D * 3    * sizeof(char));
	_near1d_dir     = (char *)        malloc(_MAXNEAR1D        * sizeof(char));
	_near1d_pos     = (double (*)[2]) malloc(_MAXNEAR1D * 2    * sizeof(double));

	_near2d_cmp     = (char (*)[3])   malloc(_MAXNEAR2D * 3    * sizeof(char));
	_near2d_dir     = (char *)        malloc(_MAXNEAR2D        * sizeof(char));
	_near2d_pos     = (double *)      malloc(_MAXNEAR2D        * sizeof(double));
}


void ofd_section_size(int size)
{
	_MAXSECTION  = size;
	_xsection    = (double *)         realloc(_xsection,       (_MAXSECTION + 1)  * sizeof(double));
	_ysection    = (double *)         realloc(_ysection,       (_MAXSECTION + 1)  * sizeof(double));
	_zsection    = (double *)         realloc(_zsection,       (_MAXSECTION + 1)  * sizeof(double));
	_xdivision   = (int *)            realloc(_xdivision,       _MAXSECTION       * sizeof(int));
	_ydivision   = (int *)            realloc(_ydivision,       _MAXSECTION       * sizeof(int));
	_zdivision   = (int *)            realloc(_zdivision,       _MAXSECTION       * sizeof(int));
}
void ofd_material_size(int size)
{
	// size > 256 => compile with "/D_ID16",...
	_MAXMATERIAL    = size;
	_material       = (double (*)[4]) realloc(_material,       _MAXMATERIAL * 4  * sizeof(double));
	_material_type  = (int *)         realloc(_material_type,  _MAXMATERIAL      * sizeof(int));
	_material_name  = (char **)       realloc(_material_name,  _MAXMATERIAL      * sizeof(char *));
}
void ofd_geometry_size(int size)
{
	_MAXGEOMETRY    = size;
	_geometry_m     = (int *)         realloc(_geometry_m,     _MAXGEOMETRY      * sizeof(int));
	_geometry_shape = (int *)         realloc(_geometry_shape, _MAXGEOMETRY      * sizeof(int));
	_geometry_name  = (char **)       realloc(_geometry_name,  _MAXGEOMETRY      * sizeof(char *));
	_geometry       = (double (*)[8]) realloc(_geometry,       _MAXGEOMETRY * 8  * sizeof(double));
}
void ofd_feed_size(int size)
{
	_MAXFEED        = size;
	_feed_dir       = (char *)        realloc(_feed_dir,       _MAXFEED          * sizeof(char));
	_feed_par       = (double (*)[6]) realloc(_feed_par,       _MAXFEED * 6      * sizeof(double));
}
void ofd_load_size(int size)
{
	_MAXLOAD        = size;
	_load_dir       = (char *)        realloc(_load_dir,       _MAXLOAD          * sizeof(char));
	_load_type      = (char *)        realloc(_load_type,      _MAXLOAD          * sizeof(char));
	_load_par       = (double (*)[4]) realloc(_load_par,       _MAXLOAD * 4      * sizeof(double));
}
void ofd_point_size(int size)
{
	_MAXPOINT       = size;
	_point_dir      = (char *)        realloc(_point_dir,      _MAXPOINT         * sizeof(char));
	_point_pos      = (double (*)[3]) realloc(_point_pos,      _MAXPOINT * 3     * sizeof(double));
}
void ofd_far1d_size(int size)
{
	_MAXFAR1D       = size;
	_far1d_dir      = (char *)        realloc(_far1d_dir,      _MAXFAR1D         * sizeof(char));
	_far1d_div      = (int *)         realloc(_far1d_div,      _MAXFAR1D         * sizeof(int));
	_far1d_angle    = (double *)      realloc(_far1d_angle,    _MAXFAR1D         * sizeof(double));
}
void ofd_near1d_size(int size)
{
	_MAXNEAR1D      = size;
	_near1d_cmp     = (char (*)[3])   realloc(_near1d_cmp,     _MAXNEAR1D * 3    * sizeof(char));
	_near1d_dir     = (char *)        realloc(_near1d_dir,     _MAXNEAR1D        * sizeof(char));
	_near1d_pos     = (double (*)[2]) realloc(_near1d_pos,     _MAXNEAR1D * 2    * sizeof(double));
}
void ofd_near2d_size(int size)
{
	_MAXNEAR2D      = size;
	_near2d_cmp     = (char (*)[3])   realloc(_near2d_cmp,     _MAXNEAR2D * 3    * sizeof(char));
	_near2d_dir     = (char *)        realloc(_near2d_dir,     _MAXNEAR2D        * sizeof(char));
	_near2d_pos     = (double *)      realloc(_near2d_pos,     _MAXNEAR2D        * sizeof(double));
}


void ofd_title(const char title[]) {strcpy(_title, title);}


void ofd_xsection1(double x) {_xsection[_xsections++] = x;}
void ofd_ysection1(double y) {_ysection[_ysections++] = y;}
void ofd_zsection1(double z) {_zsection[_zsections++] = z;}
void ofd_xdivision1(int n) {_xdivision[_xdivisions++] = (n != 0) ? n : 1;}
void ofd_ydivision1(int n) {_ydivision[_ydivisions++] = (n != 0) ? n : 1;}
void ofd_zdivision1(int n) {_zdivision[_zdivisions++] = (n != 0) ? n : 1;}


void ofd_xsection(int n, ...)
{
	int i;
	va_list ap;
	va_start(ap, n);
	for (i = 0; i < n; i++) {
		_xsection[i] = va_arg(ap, double);
	}
	va_end(ap);
	_xsections = n;
}


void ofd_ysection(int n, ...)
{
	int i;
	va_list ap;
	va_start(ap, n);
	for (i = 0; i < n; i++) {
		_ysection[i] = va_arg(ap, double);
	}
	va_end(ap);
	_ysections = n;
}


void ofd_zsection(int n, ...)
{
	int i;
	va_list ap;
	va_start(ap, n);
	for (i = 0; i < n; i++) {
		_zsection[i] = va_arg(ap, double);
	}
	va_end(ap);
	_zsections = n;
}


void ofd_xdivision(int n, ...)
{
	int i;
	va_list ap;
	va_start(ap, n);
	for (i = 0; i < n; i++) {
		_xdivision[i] = va_arg(ap, int);
		if (_xdivision[i] == 0) _xdivision[i] = 1;
	}
	va_end(ap);
	_xdivisions = n;
}


void ofd_ydivision(int n, ...)
{
	int i;
	va_list ap;
	va_start(ap, n);
	for (i = 0; i < n; i++) {
		_ydivision[i] = va_arg(ap, int);
		if (_ydivision[i] == 0) _ydivision[i] = 1;
	}
	va_end(ap);
	_ydivisions = n;
}


void ofd_zdivision(int n, ...)
{
	int i;
	va_list ap;
	va_start(ap, n);
	for (i = 0; i < n; i++) {
		_zdivision[i] = va_arg(ap, int);
		if (_zdivision[i] == 0) _zdivision[i] = 1;
	}
	va_end(ap);
	_zdivisions = n;
}


static void _ofd_material(int type, double d1, double d2, double d3, double d4, const char name[])
{
	if (_materials >= _MAXMATERIAL) {fprintf(stderr, "*** too many materials > %d\n", _MAXMATERIAL); exit(1);}

	_material_type[_materials] = type;
	_material[_materials][0] = d1;
	_material[_materials][1] = d2;
	_material[_materials][2] = d3;
	_material[_materials][3] = d4;
	if (strcmp(name, "")) {
		_material_name[_materials] = (char *)malloc((strlen(name) + 1) * sizeof(char));
		strcpy(_material_name[_materials], name);
	}
	else {
		_material_name[_materials] = NULL;
	}
	_materials++;
}


void ofd_material(double epsr, double esgm, double amur, double msgm, const char name[])
{
	_ofd_material(1, epsr, esgm, amur, msgm, name);
}


void ofd_material_dispersion(double einf, double ae, double be, double ce, const char name[])
{
	_ofd_material(2, einf, ae, be, ce, name);
}


void ofd_geometry_array(int material, int shape, const double g[])
{
	if (_geometrys >= _MAXGEOMETRY) {fprintf(stderr, "*** too many geometries > %d\n", _MAXGEOMETRY); exit(1);}

	int narray = ((shape == 31) || (shape == 32) || (shape == 33)
	           || (shape == 41) || (shape == 42) || (shape == 43)
	           || (shape == 51) || (shape == 52) || (shape == 53)) ? 8 : 6;
	_geometry_m[_geometrys] = material;
	_geometry_shape[_geometrys] = shape;
	memcpy(_geometry[_geometrys], g, narray * sizeof(double));
	_geometry_name[_geometrys] = NULL;
	_geometrys++;
}


void ofd_geometry(int material, int shape, double x1, double x2, double y1, double y2, double z1, double z2)
{
	double g[8];
	g[0] = x1;
	g[1] = x2;
	g[2] = y1;
	g[3] = y2;
	g[4] = z1;
	g[5] = z2;
	g[6] =
	g[7] = 0;

	ofd_geometry_array(material, shape, g);
}


void ofd_geometry_name(const char name[])
{
	if (strcmp(name, "") && (_geometrys > 0)) {
		_geometry_name[_geometrys - 1] = (char *)malloc((strlen(name) + 1) * sizeof(char));
		strcpy(_geometry_name[_geometrys - 1], name);
	}
}


void ofd_feed(char dir, double x, double y, double z, double amp, double delay, double z0)
{
	if (_feeds >= _MAXFEED) {fprintf(stderr, "*** too many feeds > %d\n", _MAXFEED); exit(1);}

	_feed_dir[_feeds] = dir;
	_feed_par[_feeds][0] = x;
	_feed_par[_feeds][1] = y;
	_feed_par[_feeds][2] = z;
	_feed_par[_feeds][3] = amp;
	_feed_par[_feeds][4] = delay;
	_feed_par[_feeds][5] = z0;
	_feeds++;
}


void ofd_planewave(double theta, double phi, int pol)
{
	_planewave = 1;
	_planewave_theta = theta;
	_planewave_phi   = phi;
	_planewave_pol   = pol;
}


void ofd_load(char dir, double x, double y, double z, char type, double rcl)
{
	if (_loads >= _MAXLOAD) {fprintf(stderr, "*** too many loads > %d\n", _MAXLOAD); exit(1);}

	_load_dir[_loads] = dir;
	_load_type[_loads] = type;
	_load_par[_loads][0] = x;
	_load_par[_loads][1] = y;
	_load_par[_loads][2] = z;
	_load_par[_loads][3] = rcl;
	_loads++;
}


void ofd_point(char dir, double x, double y, double z, const char prop[])
{
	if (_points >= _MAXPOINT) {fprintf(stderr, "*** too many points > %d\n", _MAXPOINT); exit(1);}

	_point_dir[_points] = dir;
	_point_pos[_points][0] = x;
	_point_pos[_points][1] = y;
	_point_pos[_points][2] = z;
	if (_points == 0) {
		strcpy(_point_p, prop);
	}
	_points++;
}


void ofd_rfeed(double rfeed) {_rfeed = rfeed;}
void ofd_pulsewidth(double pulsewidth) {_pulsewidth = pulsewidth;}
void ofd_timestep(double timestep) {_timestep = timestep;}


void ofd_pml(int l, double m, double r0)
{
	_abc = 1;
	_pml_l = l;
	_pml_m = m;
	_pml_r0 = r0;
}


void ofd_pbc(int pbcx, int pbcy, int pbcz)
{
	_pbcx = pbcx;
	_pbcy = pbcy;
	_pbcz = pbcz;
}


void ofd_frequency1(double fstart, double fend, int div)
{
	_f1start = fstart;
	_f1end   = fend;
	_f1div   = div;
}


void ofd_frequency2(double fstart, double fend, int div)
{
	_f2start = fstart;
	_f2end   = fend;
	_f2div   = div;
}


void ofd_solver(int maxiter, int nout, double converg)
{
	_maxiter = maxiter;
	_nout    = nout;
	_converg = converg;
}

// post

void ofd_matchingloss(int i0)
{
	_matchingloss = i0;
}


void ofd_plotiter(int i0)
{
	_plotiter = i0;
}


void ofd_plotfeed(int i0)
{
	_plotfeed = i0;
}


void ofd_plotpoint(int i0)
{
	_plotpoint = i0;
}


void ofd_plotsmith(int i0)
{
	_plotsmith = i0;
}


void ofd_plotzin(int scale, double min, double max, int div)
{
	if      (scale == 1) {
		_plotzin = 1;
		_zinscale.user = 0;
	}
	else if (scale == 2) {
		_plotzin = 1;
		_zinscale.user = 1;
		_zinscale.min = min;
		_zinscale.max = max;
		_zinscale.div = div;
	}
}


void ofd_plotyin(int scale, double min, double max, int div)
{
	if      (scale == 1) {
		_plotyin = 1;
		_yinscale.user = 0;
	}
	else if (scale == 2) {
		_plotyin = 1;
		_yinscale.user = 1;
		_yinscale.min = min;
		_yinscale.max = max;
		_yinscale.div = div;
	}
}


void ofd_plotref(int scale, double min, double max, int div)
{
	if      (scale == 1) {
		_plotref = 1;
		_refscale.user = 0;
	}
	else if (scale == 2) {
		_plotref = 1;
		_refscale.user = 1;
		_refscale.min = min;
		_refscale.max = max;
		_refscale.div = div;
	}
}


void ofd_plotspara(int scale, double min, double max, int div)
{
	if      (scale == 1) {
		_plotspara = 1;
		_sparascale.user = 0;
	}
	else if (scale == 2) {
		_plotspara = 1;
		_sparascale.user = 1;
		_sparascale.min = min;
		_sparascale.max = max;
		_sparascale.div = div;
	}
}


void ofd_plotcoupling(int scale, double min, double max, int div)
{
	if      (scale == 1) {
		_plotcoupling = 1;
		_couplingscale.user = 0;
	}
	else if (scale == 2) {
		_plotcoupling = 1;
		_couplingscale.user = 1;
		_couplingscale.min = min;
		_couplingscale.max = max;
		_couplingscale.div = div;
	}
}


void ofd_plotfar0d(double theta, double phi, int scale, double min, double max, int div)
{
	if      (scale == 1) {
		_plotfar0d = 1;
		_far0d_angle[0] = theta;
		_far0d_angle[1] = phi;
		_far0dscale.user = 0;
	}
	else if (scale == 2) {
		_plotfar0d = 1;
		_far0d_angle[0] = theta;
		_far0d_angle[1] = phi;
		_far0dscale.user = 1;
		_far0dscale.min = min;
		_far0dscale.max = max;
		_far0dscale.div = div;
	}
}


void ofd_freqdiv(int freqdiv)
{
	_freqdiv = freqdiv;
}


void ofd_plotfar1d(char dir, int div, double angle)
{
	if (_far1ds >= _MAXFAR1D) {fprintf(stderr, "*** too many far1ds > %d\n", _MAXFAR1D); exit(1);}

	_far1d_dir[_far1ds] = dir;
	_far1d_div[_far1ds] = div;
	_far1d_angle[_far1ds] = angle;
	_far1ds++;
}


void ofd_far1dstyle(int i0)
{
	_far1dstyle = i0;
}


void ofd_far1dcomponent(int i0, int i1, int i2)
{
	_far1dcomponent[0] = i0;
	_far1dcomponent[1] = i1;
	_far1dcomponent[2] = i2;
}


void ofd_far1ddb(int i0)
{
	_far1dscale.db = (i0 == 0) ? 0 : 1;
}


void ofd_far1dnorm(int i0)
{
	_far1dnorm = i0;
}


void ofd_far1dscale(double min, double max, int div)
{
	_far1dscale.user = 1;
	_far1dscale.min = min;
	_far1dscale.max = max;
	_far1dscale.div = div;
}


void ofd_plotfar2d(int divtheta, int divphi)
{
	_far2ds = 1;
	_far2d_div[0] = divtheta;
	_far2d_div[1] = divphi;
}


void ofd_far2dcomponent(int i0, int i1, int i2, int i3, int i4, int i5, int i6)
{
	_far2dcomponent[0] = i0;
	_far2dcomponent[1] = i1;
	_far2dcomponent[2] = i2;
	_far2dcomponent[3] = i3;
	_far2dcomponent[4] = i4;
	_far2dcomponent[5] = i5;
	_far2dcomponent[6] = i6;
}


void ofd_far2ddb(int i0)
{
	_far2dscale.db = (i0 == 0) ? 0 : 1;
}


void ofd_far2dscale(double min, double max, int div)
{
	_far2dscale.user = 1;
	_far2dscale.min = min;
	_far2dscale.max = max;
	_far2dscale.div = div;
}


void ofd_far2dobj(double obj)
{
	_far2dobj = obj;
}


void ofd_plotnear1d(const char component[], char dir, double p1, double p2)
{
	if (_near1ds >= _MAXNEAR1D) {fprintf(stderr, "*** too many near1d.s > %d\n", _MAXNEAR1D); exit(1);}

	strcpy(_near1d_cmp[_near1ds], component);
	_near1d_dir[_near1ds] = dir;
	_near1d_pos[_near1ds][0] = p1;
	_near1d_pos[_near1ds][1] = p2;
	_near1ds++;
}


void ofd_near1ddb(int i0)
{
	_near1dscale.db = (i0 == 0) ? 0 : 1;
}


void ofd_near1dnoinc(int i0)
{
	_near1dnoinc = i0;
}


void ofd_near1dscale(double min, double max, int div)
{
	_near1dscale.user = 1;
	_near1dscale.min = min;
	_near1dscale.max = max;
	_near1dscale.div = div;
}


void ofd_plotnear2d(const char component[], char dir, double p)
{
	if (_near2ds >= _MAXNEAR2D) {fprintf(stderr, "*** too many near2d.s > %d\n", _MAXNEAR2D); exit(1);}

	strcpy(_near2d_cmp[_near2ds], component);
	_near2d_dir[_near2ds] = dir;
	_near2d_pos[_near2ds] = p;
	_near2ds++;
}


void ofd_near2ddim(int i0, int i1)
{
	_near2ddim[0] = i0;
	_near2ddim[1] = i1;
}


void ofd_near2dframe(int i0)
{
	_near2dframe = i0;
}


void ofd_near2ddb(int i0)
{
	_near2dscale.db = (i0 == 0) ? 0 : 1;
}


void ofd_near2dscale(double min, double max)
{
	_near2dscale.user = 1;
	_near2dscale.min = min;
	_near2dscale.max = max;
}


void ofd_near2dcontour(int i0)
{
	_near2dcontour = i0;
}


void ofd_near2dobj(int i0)
{
	_near2dobj = i0;
}


void ofd_near2dnoinc(int i0)
{
	_near2dnoinc = i0;
}


void ofd_near2dzoom(double p0, double p1, double p2, double p3)
{
	_near2dizoom = 1;
	_near2dpzoom[0] = p0;
	_near2dpzoom[1] = p1;
	_near2dpzoom[2] = p2;
	_near2dpzoom[3] = p3;
}


void ofd_outdata(const char fn[])
{
	FILE  *fp;
	if ((fp = fopen(fn, "w")) == NULL) {fprintf(stderr, "output file open error : %s\n", fn); exit(1);}

	fprintf(fp, "%s\n", PROGRAM);

	if (strcmp(_title, "")) {
		fprintf(fp, "title = %s\n", _title);
	}

	if (_xsections != _xdivisions + 1) {printf("invalid X-sections. (%d, %d)\n", _xsections, _xdivisions + 1); exit(1);}
	fprintf(fp, "%s", "xmesh = ");
	for (int i = 0; i < _xdivisions; i++) {
		fprintf(fp, "%g %d ", _xsection[i], _xdivision[i]);
	}
	fprintf(fp, "%g\n", _xsection[_xdivisions]);

	if (_ysections != _ydivisions + 1) {printf("invalid Y-sections. (%d, %d)\n", _ysections, _ydivisions + 1); exit(1);}
	fprintf(fp, "%s", "ymesh = ");
	for (int j = 0; j < _ydivisions; j++) {
		fprintf(fp, "%g %d ", _ysection[j], _ydivision[j]);
	}
	fprintf(fp, "%g\n", _ysection[_ydivisions]);

	if (_zsections != _zdivisions + 1) {printf("invalid Z-sections. (%d, %d)\n", _zsections, _zdivisions + 1); exit(1);}
	fprintf(fp, "%s", "zmesh = ");
	for (int k = 0; k < _zdivisions; k++) {
		fprintf(fp, "%g %d ", _zsection[k], _zdivision[k]);
	}
	fprintf(fp, "%g\n", _zsection[_zdivisions]);

	for (int n = 0; n < _materials; n++) {
		fprintf(fp, "material = %d %g %g %g %g",
			_material_type[n], _material[n][0], _material[n][1], _material[n][2], _material[n][3]);
		if (_material_name[n] != NULL) {
			fprintf(fp, " %s", _material_name[n]);
		}
		fprintf(fp, "\n");
	}

	for (int n = 0; n < _geometrys; n++) {
		fprintf(fp, "geometry = %d %d %g %g %g %g %g %g",
			_geometry_m[n], _geometry_shape[n],
			_geometry[n][0], _geometry[n][1], _geometry[n][2],
			_geometry[n][3], _geometry[n][4], _geometry[n][5]);
		if ((_geometry_shape[n] == 31) || (_geometry_shape[n] == 32) || (_geometry_shape[n] == 33)
		 || (_geometry_shape[n] == 41) || (_geometry_shape[n] == 42) || (_geometry_shape[n] == 43)
		 || (_geometry_shape[n] == 51) || (_geometry_shape[n] == 52) || (_geometry_shape[n] == 53)) {
			fprintf(fp, " %g %g", _geometry[n][6], _geometry[n][7]);
		}
		fprintf(fp, "\n");
		if (_geometry_name[n] != NULL) {
			fprintf(fp, "name = %s\n", _geometry_name[n]);
		}
	}

	for (int n = 0; n < _feeds; n++) {
		fprintf(fp, "feed = %c %g %g %g %g %g %g\n",
			_feed_dir[n], _feed_par[n][0], _feed_par[n][1], _feed_par[n][2],
			              _feed_par[n][3], _feed_par[n][4], _feed_par[n][5]);
	}

	if (_planewave) {
		fprintf(fp, "planewave = %g %g %d\n", _planewave_theta, _planewave_phi, _planewave_pol);
	}

	for (int n = 0; n < _loads; n++) {
		fprintf(fp, "load = %c %g %g %g %c %g\n",
			_load_dir[n], _load_par[n][0], _load_par[n][1], _load_par[n][2],
			              _load_type[n], _load_par[n][3]);
	}

	if (_points > 0) {
		fprintf(fp, "point = %c %g %g %g %s\n",
			_point_dir[0], _point_pos[0][0], _point_pos[0][1], _point_pos[0][2], _point_p);
		for (int n = 1; n < _points; n++) {
			fprintf(fp, "point = %c %g %g %g\n",
				_point_dir[n], _point_pos[n][0], _point_pos[n][1], _point_pos[n][2]);
		}
	}

	if (_rfeed > 1e-6) {
		fprintf(fp, "rfeed = %g\n", _rfeed);
	}

	if (_abc == 1) {
		fprintf(fp, "abc = %d %d %g %g\n", _abc, _pml_l, _pml_m, _pml_r0);
	}

	if (_pbcx || _pbcy || _pbcz) {
		fprintf(fp, "pbc = %d %d %d\n", _pbcx, _pbcy, _pbcz);
	}

	fprintf(fp, "frequency1 = %g %g %d\n", _f1start, _f1end, _f1div);
	fprintf(fp, "frequency2 = %g %g %d\n", _f2start, _f2end, _f2div);

	if (_maxiter || _nout) {
		fprintf(fp, "solver = %d %d %g\n", _maxiter, _nout, _converg);
	}

	if (_timestep > 1e-20) {
		fprintf(fp, "timestep = %g\n", _timestep);
	}
	if (_pulsewidth > 1e-20) {
		fprintf(fp, "pulsewidth = %g\n", _pulsewidth);
	}

	// post

	if (_matchingloss) {
		fprintf(fp, "matchingloss = 1\n");
	}

	if (_plotiter) {
		fprintf(fp, "plotiter = %d\n",  _plotiter);
	}

	if (_plotfeed) {
		fprintf(fp, "plotfeed = %d\n",  _plotfeed);
	}

	if (_plotpoint) {
		fprintf(fp, "plotpoint = %d\n", _plotpoint);
	}

	if (_plotsmith) {
		fprintf(fp, "plotsmith = 1\n");
	}

	if (_plotzin) {
		if      (_zinscale.user == 0) {
			fprintf(fp, "plotzin = 1\n");
		}
		else if (_zinscale.user == 1) {
			fprintf(fp, "plotzin = 2 %g %g %d\n", _zinscale.min, _zinscale.max, _zinscale.div);
		}
	}

	if (_plotyin) {
		if      (_yinscale.user == 0) {
			fprintf(fp, "plotyin = 1\n");
		}
		else if (_yinscale.user == 1) {
			fprintf(fp, "plotyin = 2 %g %g %d\n", _yinscale.min, _yinscale.max, _yinscale.div);
		}
	}

	if (_plotref) {
		if      (_refscale.user == 0) {
			fprintf(fp, "plotref = 1\n");
		}
		else if (_refscale.user == 1) {
			fprintf(fp, "plotref = 2 %g %g %d\n", _refscale.min, _refscale.max, _refscale.div);
		}
	}

	if (_plotspara) {
		if      (_sparascale.user == 0) {
			fprintf(fp, "plotspara = 1\n");
		}
		else if (_sparascale.user == 1) {
			fprintf(fp, "plotspara = 2 %g %g %d\n", _sparascale.min, _sparascale.max, _sparascale.div);
		}
	}

	if (_plotcoupling) {
		if      (_couplingscale.user == 0) {
			fprintf(fp, "plotcoupling = 1\n");
		}
		else if (_couplingscale.user == 1) {
			fprintf(fp, "plotcoupling = 2 %g %g %d\n", _couplingscale.min, _couplingscale.max, _couplingscale.div);
		}
	}

	if (_plotfar0d) {
		if      (_far0dscale.user == 0) {
			fprintf(fp, "plotfar0d = %g %g 1\n", _far0d_angle[0], _far0d_angle[1]);
		}
		else if (_far0dscale.user == 1) {
			fprintf(fp, "plotfar0d = %g %g 2 %g %g %d\n", _far0d_angle[0], _far0d_angle[1], _far0dscale.min, _far0dscale.max, _far0dscale.div);
		}
	}

	if (_freqdiv != 10) {  // default = 10
		fprintf(fp, "freqdiv = %d\n", _freqdiv);
	}

	if (_far1ds) {
		for (int n = 0; n < _far1ds; n++) {
			fprintf(fp, "plotfar1d = %c %d", _far1d_dir[n], _far1d_div[n]);
			if ((_far1d_dir[n] == 'V') || (_far1d_dir[n] == 'H')) {
				fprintf(fp, " %g", _far1d_angle[n]);
			}
			fprintf(fp, "%s", "\n");
		}
		fprintf(fp, "far1dstyle = %d\n", _far1dstyle);
		fprintf(fp, "far1dcomponent = %d %d %d\n", _far1dcomponent[0], _far1dcomponent[1], _far1dcomponent[2]);
		fprintf(fp, "far1ddb = %d\n", _far1dscale.db);
		if (_far1dscale.user) {
			fprintf(fp, "far1dscale = %g %g %d\n", _far1dscale.min, _far1dscale.max, _far1dscale.div);
		}
		if (_far1dnorm) {
			fprintf(fp, "far1dnorm = 1\n");
		}
	}

	if (_far2ds) {
		fprintf(fp, "plotfar2d = %d %d\n", _far2d_div[0], _far2d_div[1]);
		fprintf(fp, "far2dcomponent = %d %d %d %d %d %d %d\n", _far2dcomponent[0], _far2dcomponent[1], _far2dcomponent[2], _far2dcomponent[3], _far2dcomponent[4], _far2dcomponent[5], _far2dcomponent[6]);
		fprintf(fp, "far2ddb = %d\n", _far2dscale.db);
		if (_far2dscale.user) {
			fprintf(fp, "far2dscale = %g %g %d\n", _far2dscale.min, _far2dscale.max, _far2dscale.div);
		}
		fprintf(fp, "far2dobj = %g\n", _far2dobj);
	}

	if (_near1ds) {
		for (int n = 0; n < _near1ds; n++) {
			fprintf(fp, "plotnear1d = %s %c %g %g\n", _near1d_cmp[n], _near1d_dir[n], _near1d_pos[n][0], _near1d_pos[n][1]);
		}
		fprintf(fp, "near1ddb = %d\n", _near1dscale.db);
		if (_near1dscale.user) {
			fprintf(fp, "near1dscale = %g %g %d\n", _near1dscale.min, _near1dscale.max, _near1dscale.div);
		}
		fprintf(fp, "near1dnoinc = %d\n", _near1dnoinc);
	}

	if (_near2ds) {
		for (int n = 0; n < _near2ds; n++) {
			fprintf(fp, "plotnear2d = %s %c %g\n", _near2d_cmp[n], _near2d_dir[n], _near2d_pos[n]);
		}
		fprintf(fp, "near2ddim = %d %d\n", _near2ddim[0], _near2ddim[1]);
		if (_near2dframe) {
			fprintf(fp, "near2dframe = %d\n", _near2dframe);
		}
		fprintf(fp, "near2ddb = %d\n", _near2dscale.db);
		if (_near2dscale.user) {
			fprintf(fp, "near2dscale = %g %g\n", _near2dscale.min, _near2dscale.max);
		}
		fprintf(fp, "near2dcontour = %d\n", _near2dcontour);
		fprintf(fp, "near2dobj = %d\n", _near2dobj);
		fprintf(fp, "near2dnoinc = %d\n", _near2dnoinc);
		if (_near2dizoom) {
			fprintf(fp, "near2dzoom = %g %g %g %g\n", _near2dpzoom[0], _near2dpzoom[1], _near2dpzoom[2], _near2dpzoom[3]);
		}
	}

	fprintf(fp, "end\n");

	fclose(fp);

	printf("output file -> %s\n", fn);
}
