#!/usr/bin/env python
"""
Score diarization system output.

This code has been modified from the DSCORE repo: https://github.com/nryant/dscore/blob/master/LICENSE. 
All copyright lies with the original authors. Please read their license before redistribution. 

"""
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
import sys
import pdb

sys.path.append('.')

# from tabulate import tabulate

from mmsc.utils.scorelib import __version__ as VERSION
from mmsc.utils.scorelib.argparse import ArgumentParser
from mmsc.utils.scorelib.rttm import load_rttm
from mmsc.utils.scorelib.turn import merge_turns, trim_turns
from mmsc.utils.scorelib.score import score
from mmsc.utils.scorelib.six import iterkeys
from mmsc.utils.scorelib.uem import gen_uem, load_uem
from mmsc.utils.scorelib.utils import error, info, warn, xor

class RefRTTMAction(argparse.Action):
    """Custom action to ensure that reference files are specified from a
    script file or from the command line but not both.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        if not xor(namespace.ref_rttm_fns, namespace.ref_rttm_scpf):
            parser.error('Exactly one of -r and -R must be set.')


class SysRTTMAction(argparse.Action):
    """Custom action to ensure that system files are specified from a script
    file or from the command line but not both.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        if not xor(namespace.sys_rttm_fns, namespace.sys_rttm_scpf):
            parser.error('Exactly one of -s and -S must be set.')


def load_rttms(rttm_fns):
    """Load speaker turns from RTTM files.

    Parameters
    ----------
    rttm_fns : list of str
        Paths to RTTM files.

    Returns
    -------
    turns : list of Turn
        Speaker turns.

    file_ids : set
        File ids found in ``rttm_fns``.
    """
    turns = []
    file_ids = set()
    for rttm_fn in rttm_fns:
        if not os.path.exists(rttm_fn):
            error('Unable to open RTTM file: %s' % rttm_fn)
            sys.exit(1)
        try:
            turns_, _, file_ids_ = load_rttm(rttm_fn)
            turns.extend(turns_)
            file_ids.update(file_ids_)
        except IOError as e:
            error('Invalid RTTM file: %s. %s' % (rttm_fn, e))
            sys.exit(1)
    return turns, file_ids


def check_for_empty_files(ref_turns, sys_turns, uem):
    """Warn on files in UEM without reference or speaker turns."""
    ref_file_ids = {turn.file_id for turn in ref_turns}
    sys_file_ids = {turn.file_id for turn in sys_turns}
    for file_id in sorted(iterkeys(uem)):
        if file_id not in ref_file_ids:
            warn('File "%s" missing in reference RTTMs.' % file_id)
        if file_id not in sys_file_ids:
            warn('File "%s" missing in system RTTMs.' % file_id)
    # TODO: Clarify below warnings; this indicates that there are no
    #       ELIGIBLE reference/system turns.
    if not ref_turns:
        warn('No reference speaker turns found within UEM scoring regions.')
    if not sys_turns:
        warn('No system speaker turns found within UEM scoring regions.')


def load_script_file(fn):
    """Load file names from ``fn``."""
    with open(fn, 'rb') as f:
        return [line.decode('utf-8').strip() for line in f]


def print_output(file_scores, global_scores, n_digits=2):
    """Print outputs. 

    Parameters
    ----------
    file_to_scores : dict
        Mapping from file ids in ``uem`` to ``Scores`` instances.

    global_scores : Scores
        Global scores.

    n_digits : int, optional
        Number of decimal digits to display.
        (Default: 3)


    """
    
    # You can print individual file scores using the file_scores dict 
    print("Primary Metric DER=" + str(global_scores[1]))
    print("JER=" + str(global_scores[2]))

def metrics(sys_rttm_fns ,ref_rttm_fns):
    ref_turns, _ = load_rttms(ref_rttm_fns)
    sys_turns, _ = load_rttms(sys_rttm_fns)
    uem = gen_uem(ref_turns, sys_turns)
    ref_turns = trim_turns(ref_turns, uem)
    sys_turns = trim_turns(sys_turns, uem)
    ref_turns = merge_turns(ref_turns)
    sys_turns = merge_turns(sys_turns)
    check_for_empty_files(ref_turns, sys_turns, uem)
    
    return score(
        ref_turns, sys_turns, uem, step=0.010,
        jer_min_ref_dur=0.0, collar=0.25,
        ignore_overlaps=False)

def main():
    """Main."""
    # Parse command line arguments.
    parser = ArgumentParser(
        description='Score diarization from RTTM files.', add_help=True,
        usage='%(prog)s [options]')
    parser.add_argument(
        '-r', nargs='+', default=[], metavar='STR', dest='ref_rttm_fns',
        action=RefRTTMAction,
        help='reference RTTM files (default: %(default)s)')
    parser.add_argument(
        '-R', nargs=None, metavar='STR', dest='ref_rttm_scpf',
        action=RefRTTMAction,
        help='reference RTTM script file (default: %(default)s)')
    parser.add_argument(
        '-s', nargs='+', default=[], metavar='STR', dest='sys_rttm_fns',
        action=SysRTTMAction,
        help='system RTTM files (default: %(default)s)')
    parser.add_argument(
        '-S', nargs=None, metavar='STR', dest='sys_rttm_scpf',
        action=SysRTTMAction,
        help='system RTTM script file (default: %(default)s)')
    parser.add_argument(
        '-u,--uem', nargs=None, metavar='STR', dest='uemf',
        help='un-partitioned functions.evaluation map file (default: %(default)s)')
    parser.add_argument(
        '--collar', nargs=None, default=0.25, type=float, metavar='FLOAT',
        help='collar size in seconds for DER computaton '
             '(default: %(default)s)')
    parser.add_argument(
        '--ignore_overlaps', action='store_true', default=False,
        help='ignore overlaps when computing DER')
    parser.add_argument(
        '--jer_min_ref_dur', nargs=None, default=0.0, metavar='FLOAT',
        help='minimum reference speaker duration for JER '
        '(default: %(default)s)')
    parser.add_argument(
        '--step', nargs=None, default=0.010, type=float, metavar='FLOAT',
        help='step size in seconds (default: %(default)s)')
    parser.add_argument(
        '--n_digits', nargs=None, default=2, type=int, metavar='INT',
        help='number of decimal places to print (default: %(default)s)')

    parser.add_argument(
        '--version', action='version',
        version='%(prog)s ' + VERSION)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # Check that at least one reference RTTM and at least one system RTTM
    # was specified.
    if args.ref_rttm_scpf is not None:
        args.ref_rttm_fns = load_script_file(args.ref_rttm_scpf)
    if args.sys_rttm_scpf is not None:
        args.sys_rttm_fns = load_script_file(args.sys_rttm_scpf)
    if not args.ref_rttm_fns:
        error('No reference RTTMs specified.')
        sys.exit(1)
    if not args.sys_rttm_fns:
        error('No system RTTMs specified.')
        sys.exit(1)

    # Load speaker/reference speaker turns and UEM. If no UEM specified,
    # determine it automatically.
    info('Loading speaker turns from reference RTTMs...', file=sys.stderr)
    ref_turns, _ = load_rttms(args.ref_rttm_fns)
    info('Loading speaker turns from system RTTMs...', file=sys.stderr)
    sys_turns, _ = load_rttms(args.sys_rttm_fns)
    if args.uemf is not None:
        info('Loading universal functions.evaluation map...', file=sys.stderr)
        uem = load_uem(args.uemf)
    else:
        warn('No universal functions.evaluation map specified. Approximating from '
             'reference and speaker turn extents...')
        uem = gen_uem(ref_turns, sys_turns)

    # Trim turns to UEM scoring regions and merge any that overlap.
    info('Trimming reference speaker turns to UEM scoring regions...',
         file=sys.stderr)
    ref_turns = trim_turns(ref_turns, uem)
    info('Trimming system speaker turns to UEM scoring regions...',
         file=sys.stderr)
    sys_turns = trim_turns(sys_turns, uem)
    info('Checking for overlapping reference speaker turns...',
         file=sys.stderr)
    ref_turns = merge_turns(ref_turns)
    info('Checking for overlapping system speaker turns...',
         file=sys.stderr)
    sys_turns = merge_turns(sys_turns)

    # Score.
    info('Scoring...', file=sys.stderr)
    check_for_empty_files(ref_turns, sys_turns, uem)
    
    file_scores, global_scores = score(
        ref_turns, sys_turns, uem, step=args.step,
        jer_min_ref_dur=args.jer_min_ref_dur, collar=args.collar,
        ignore_overlaps=args.ignore_overlaps)
    
    print_output(
        file_scores, global_scores, args.n_digits)

if __name__ == '__main__':
    main()
