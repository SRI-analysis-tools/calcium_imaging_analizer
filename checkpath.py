
import sys
import os

from PyQt5 import QtCore, QtGui, uic

env = os.environ.copy()

env['PYTHONPATH']       = r"C:\designer_widgets" #("%s"%os.pathsep).join(sys.path)
env['PYQTDESIGNERPATH'] = r"C:\designer_plugins"

qenv = ['%s="%s"' % (name, value) for name, value in env.items()]

# Start Designer.
designer = QtCore.QProcess()
#designer.setEnvironment(qenv)

designer_bin = r"C:\Python27x64\Lib\site-packages\PyQt4\designer.exe"

designer.start(designer_bin)
designer.waitForFinished(-1)

sys.exit(designer.exitCode())

# Check if paths are right
print ("\n  Designer Env:")
for pypath in designer.environment():
    if "PYTHONPATH" in pypath:
        print (" #  ",pypath)
    if "PYQTDESIGNERPATH" in pypath:
        print (" #  ",pypath)