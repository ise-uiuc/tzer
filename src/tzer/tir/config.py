import argparse
import os
from typing import Union, Optional
from dataclasses import dataclass
from . import util


@dataclass
class Config:
    fuzzing_time_in_minutes: Optional[Union[int, float]]
    iterations: Optional[int]
    building_timeout_in_seconds: Union[int, float]
    execution_timeout_in_seconds: Union[int, float]
    tolerance: int
    max_generation_size: int
    max_node_size: int
    use_seeds: bool
    use_lemon_seeds: bool
    use_coverage: bool
    use_coverage_feedback: bool
    mutate_control_flow_with_general_purpose_mutators: bool
    record_tir: bool
    use_pass: bool
    use_none: bool
    diff_test_rounds: int
    report_folder: Optional[str]

    def __post_init__(self):
        if self.fuzzing_time_in_minutes is None and self.iterations is None \
                or self.fuzzing_time_in_minutes is not None and self.iterations is not None:
            raise AssertionError(
                'Either fuzzing time or fuzzing iterations (not both) should be provided')
        if not self.use_coverage:
            self.use_coverage_feedback = False


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Tzer configuration')
    parser.add_argument('--fuzz-time', nargs='?', type=float,
                        help='Fuzzing duration in minute.')
    parser.add_argument('--iterations', nargs='?', type=int,
                        help='Fuzzing iterations.')
    parser.add_argument('--build-timeout', nargs='?', type=float, default=2,
                        help='Timeout in second for building one TIR')
    parser.add_argument('--tolerance', nargs='?', type=int, default=5,
                        help='IR tolerance in joint fuzzing.')
    parser.add_argument('--exec-timeout', nargs='?', type=float, default=5,
                        help='Timeout in second for running one compiled TIR')
    parser.add_argument('--max-gen', nargs='?', type=int, default=50,
                        help='Max size in generation')
    parser.add_argument('--max-node-size', nargs='?', type=int, default=200,
                        help='Max node size allowed to be added to the pool')
    parser.add_argument('--diff-test-rounds', nargs='?', type=int, default=0,
                        help='Maximum rounds for differential testing')
    parser.add_argument('--report-folder', nargs='?',
                        type=str, help='Path to store fuzzing data')
    return parser


def config_from_args(args) -> Config:
    return Config(
        fuzzing_time_in_minutes=args.fuzz_time,
        iterations=args.iterations,
        building_timeout_in_seconds=args.build_timeout,
        execution_timeout_in_seconds=args.exec_timeout,
        tolerance=args.tolerance,
        max_node_size=args.max_node_size,
        max_generation_size=args.max_gen,
        use_seeds=os.getenv('NO_SEEDS') is None,
        use_lemon_seeds=os.getenv('LEMON') is not None,
        use_coverage=os.getenv('NO_COV') is None,
        use_coverage_feedback=os.getenv('NO_FEEDBACK') is None,
        record_tir=os.getenv('TIR_REC') is not None,
        use_pass=os.getenv('PASS') is not None,
        use_none=os.getenv('NONE') is not None,
        mutate_control_flow_with_general_purpose_mutators=os.getenv(
            'CONTROL') is not None,
        diff_test_rounds=args.diff_test_rounds,
        report_folder=args.report_folder,
    )
