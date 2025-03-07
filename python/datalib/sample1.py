# -*- coding: utf-8 -*-
"""
sample1.py
OpenFDTD データ作成ライブラリ (Python) サンプルプログラム (1)

使い方:
(1) 本ファイルを編集し、ofd_datalib.py と同じフォルダにおく
(2) > python sample1.py
(3) OpenFDTD入力ファイル(sample1.ofd)が出力される
"""

import ofd_datalib as ofd

ofd.title('dipole antenna')

ofd.xsection(-0.075, 30, +0.075)
ofd.ysection(-0.075, 30, +0.075)
ofd.zsection(-0.075, 10, -0.025, 11, 0.025, 10, 0.075)

#ofd.material(2.0, 0.0, 1.0, 0.0)

ofd.geometry(1, 1, 0.0, 0.0, 0.0, 0.0, -0.025, +0.025)

ofd.feed('Z', 0.0, 0.0, 0.0, 1.0, 0.0, 50)
#ofd.planewave(90, 0, 1)
#ofd.pml(5, 2, 1e-5)

ofd.frequency1(2e9, 3e9, 10)
ofd.frequency2(3e9, 3e9, 0)

ofd.solver(1000, 50, 1e-3)

ofd.plotiter(1)
ofd.plotzin(1)
ofd.plotyin(1)
ofd.plotref(1)

ofd.plotfar1d('X', 90)
#ofd.far1dstyle(2)
#ofd.far1dcomponent(1, 1, 1)
#ofd.far1ddb(1)
#ofd.far1dnorm(1)
#ofd.far1dscale(-30, 10, 4)

ofd.plotfar2d(18, 36)
#ofd.far2dcomponent(1, 0, 0, 0, 0, 0, 0)
#ofd.far2ddb(1)
#ofd.far2dscale(-20, 10, 6)
#ofd.far2dobj(0.5)

ofd.plotnear1d('E', 'Z', 0.03, 0.0)
#ofd.near1ddb(1)
#ofd.near1dnoinc(1)
#ofd.near1dscale(-30, 20, 5)

ofd.plotnear2d('E', 'X', 0.03)
#ofd.near2ddim(1, 1)
#ofd.near2dframe(20)
#ofd.near2ddb(1)
#ofd.near2dscale(-40, +20)
#ofd.near2dcontour(1)
ofd.near2dobj(1)
#ofd.near2dnoinc(1)
#ofd.near2dzoom(-0.1, +0.1, -0.1, +0.1)

ofd.output('sample1.ofd')
