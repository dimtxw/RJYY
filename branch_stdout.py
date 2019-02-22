import sys


class BranchStdout(object):
    def __init__(self, logfile):
        self._console = sys.stdout
        self._logfile = open(logfile, 'a+')

    def write(self, content):
        self._console.write(content)
        self._logfile.write(content)

    def flush(self):
        self._console.flush()
        self._logfile.flush()

    def close(self):
        self._logfile.close()
