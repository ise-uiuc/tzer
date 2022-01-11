import argparse
import os
import shlex
import select
import subprocess
import time


_CMD_ = './fuzz_me corpus seeds'
_COV_BY_TIME_NAME_ = 'cov_by_time.txt'
_START_TIMESTAMP_ = 'start_time.txt'


class Reporter:
    def __init__(self) -> None:
        with open(_COV_BY_TIME_NAME_, 'w') as f:
            f.close()

        with open(_START_TIMESTAMP_, 'w') as f:
            f.write(str(time.time()))
        self.start_time = time.perf_counter()
        with open(_COV_BY_TIME_NAME_, 'a') as f:
            f.write(str(self.start_time)+'\n')

    def record_coverage(self, cov):
        with open(_COV_BY_TIME_NAME_, 'a') as f:
            f.write(f'{time.perf_counter() - self.start_time :.2f},{cov}\n')


def clean():
    os.system('rm corpus/*')
    os.system('rm crash-*')


def parse_cov(line):
    cov = line.index('cov: ')
    ft = line.index('ft: ')
    cov_number = line[cov + 5:ft].strip()
    return int(cov_number)


def run_libfuzzer(max_time, timeout=60):
    clean()

    reporter = Reporter()
    start_time = time.time()
    take_time = 0
    max_cov = 0
    while take_time < max_time:
        p = subprocess.Popen(shlex.split(_CMD_), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.set_blocking(p.stderr.fileno(), False)
        epoll_obj = select.epoll()
        epoll_obj.register(p.stderr, select.EPOLLIN)
        wait_time = 0
        read_start_time = time.time()
        while p.poll() == None and wait_time < timeout:
            poll_result = epoll_obj.poll(0)
            if len(poll_result) > 0:
                line = p.stderr.readline().decode().strip()
                print(line)
                if 'cov:' in line:
                    cov = parse_cov(line)
                    if cov > max_cov:
                        max_cov = cov
                    reporter.record_coverage(max_cov)
                wait_time = 0
                read_start_time = time.time()
            else:
                wait_time = time.time() - read_start_time

            if time.time() - start_time > max_time:
                p.kill()
                break

        if p.poll() == None:
            print('kill process')
            p.kill()

        _, errs = p.communicate()
        for err in errs.decode().splitlines():
            print(err.strip())
            if 'cov:' in err:
                if cov > max_cov:
                    max_cov = cov
                    reporter.record_coverage(max_cov)

        reporter.record_coverage(max_cov)
        take_time = time.time() - start_time
        print(f'run libfuzzer in {take_time}s')

    print('Finish fuzzing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timeout', type=float, default=4*60*60, nargs='?', help='timeout (seconds)')
    args = parser.parse_args()
    run_libfuzzer(args.timeout)

