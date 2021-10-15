from tvm.contrib import coverage
from tzer import fuzz, template, report
from tqdm import trange

from tzer.relay_seeds import MODEL_SEEDS

target_seeds = MODEL_SEEDS[4::]

seed = target_seeds[0]
import time

_MAX_HOURS_ = 1
_MAX_BUG_FOUND_ = 20000
_MAX_STILL_ROUND_ = 10000

if __name__ == '__main__':
    reporter = report.Reporter()

    still_round = 0
    old_cov = 0
    begin = time.time()
    with trange(100000) as t:
        for i in t:
            ctx = fuzz.make_context(seed)
            try:
                template.execute_both_mode(ctx)
            except Exception as e:
                reporter.report_bug(e, ctx, message=f'{e}\n' +
                f'Passes {[f.__name__ for f in ctx.compile.relay_pass_types]}')

            reporter.record_coverage()

            if time.time() - begin > _MAX_HOURS_ * 60 * 60:
                break
            if reporter.n_bug > _MAX_BUG_FOUND_:
                break
            if coverage.get_now() == old_cov:
                still_round += 1
                if still_round >= _MAX_STILL_ROUND_:
                    break
            else:
                still_round = 0
                old_cov = coverage.get_now()

            t.set_postfix(
                coverage=coverage.get_now() / coverage.get_total(),
                bugs_found=reporter.n_bug, 
                max_hours=_MAX_HOURS_,
                still_rounds=still_round)
