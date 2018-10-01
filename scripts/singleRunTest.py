import sys
from stabilityRun import runSingleTestJob

if __name__ == "__main__":
    job_index = int(sys.argv[1])
    runSingleTestJob(job_index)