import subprocess as sb

sb.call("python line_profiler/kernprof.py -l -v Feminist_2Cell_CNN_Camera.py",shell=True)
# sb.call("python -m memory_profiler Feminist_2Cell_CNN_Camera.py",shell=True)

