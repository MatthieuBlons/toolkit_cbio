import time


# %%
class timetracker(object):
    """
    Stopwatch  timer
    """

    def __init__(self, name=None, verbose=False):
        self.name = name
        self.verbose = verbose
        self.TicToc = self.TicTocGenerator()

    def TicTocGenerator(self):  # add verbose arg
        # Generator that returns time differences
        ti = 0  # initial time
        tf = time.time()  # final time
        while True:
            ti = tf
            tf = time.time()
            yield tf - ti  # returns the time difference

    def toc(self, tempBool=True):
        tempTimeInterval = next(self.TicToc)
        if tempBool:
            if self.verbose:
                print("Elapsed time: %f seconds." % tempTimeInterval)
        return tempTimeInterval

    def tic(self):
        # Records a time in TicToc, marks the beginning of a time interval
        self.toc(False)
