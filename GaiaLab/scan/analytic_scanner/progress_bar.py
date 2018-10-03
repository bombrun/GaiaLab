# example of a simple progress bar without specific external libraries
# note that it might be useless
# found on stack overflow so the copyright is not mine (mine = Luca)

import time
import sys

toolbar_width = 50

# setup toolbar
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1))  # return to start of line, after '['

for i in xrange(toolbar_width):
    time.sleep(0.1)  # do real work here
    # update the bar
    sys.stdout.write("-")
    sys.stdout.flush()

sys.stdout.write("\n")
