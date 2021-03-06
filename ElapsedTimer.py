import time

class ElapsedTimer:
  def __init__(self):
    self.start_time = time.time()
  def elapsed(self,sec):
    if sec < 60:
      return str(sec) + " sec"
    elif sec < (60 * 60):
      return str(sec / 60) + " min"
    else:
      return str(sec / (60 * 60)) + " hr"
  def elapsedTime(self):
    print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )