"""
Runs the trials for all (ng, nc) pairs with multiprocessing.

On a 6-core 2019 MacBook Pro, this takes close to an hour and a half.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import multiprocessing as mp
import os
from typing import List

import sys

sys.path.append("../")
from common import Sphere, normalize
from fpme_utils import FairOracle, create_a_B_lamb_T, compute_B_err
from fpme import FPME
from trials import (
    NUM_TRIALS,
    load_fpme_sphere,
    load_a_B_lamb_T,
    write_fpme_trial,
    write_fpme_trial_summary,
)


SEARCH_TOL = 1e-2
NUM_PROCS = 6
QUEUE_IN = mp.Queue()
QUEUE_OUT = mp.Queue()


def run_trial(
    ng: int,
    nc: int,
    q: int,
    sphere: Sphere,
    a: np.array,
    B: List[List[np.matrix]],
    lamb: float,
    T: np.array,
):
    """
    Run an FPME trial. Inputs are hopefully, see fpme.py/FairOracle for a thorough description.
    """
    fair_oracle = FairOracle(a, B, lamb, T)
    fpm = FPME(sphere, fair_oracle, T, nc, q, ng, SEARCH_TOL)
    a_hat, B_hat, lamb_hat = fpm.run_fpme()
    return (a_hat, B_hat, lamb_hat)


def proc_run_trials(self_id: int):
    """
    A process that polls QUEUE_IN for work and puts results into QUEUE_OUT.
    Process exits when it gets None from the queue.
    """
    while True:
        data = QUEUE_IN.get(block=True)
        if data is None:
            QUEUE_IN.put(None)  # so other processes can read this and exit out
            break  # exit

        tid, ng, nc, q, sphere, a, B, lamb, T = data
        a_hat, B_hat, lamb_hat = run_trial(ng, nc, q, sphere, a, B, lamb, T)

        # put result into queue out
        QUEUE_OUT.put((tid, a_hat, B_hat, lamb_hat))


def manage_trials(ng: int, nc: int, q: int):
    """
    Run by main process. Schedules the work into QUEUE_IN and reads results
    from QUEUE_OUT. Saves results to disk.
    """
    if os.path.exists(f"trials/fpme/m={ng},k={nc}/a_1_hat.npy"):
        print("m={} k={} already run".format(ng, nc))
        return

    sphere = load_fpme_sphere(ng, nc)
    # put in work
    trial_ids = []
    a_list = []
    B_list = []
    lamb_list = []
    T_list = []
    for i in tqdm(range(NUM_TRIALS)):
        a, B, lamb, T = load_a_B_lamb_T(ng, nc, i)
        trial_ids.append(i)
        a_list.append(a)
        B_list.append(B)
        lamb_list.append(lamb)
        T_list.append(T)

        QUEUE_IN.put((i, ng, nc, q, sphere, a, B, lamb, T))

    # use trial_ids_out to map into the original inputs
    trial_ids_out = []

    a_hat_list = []
    B_hat_list = []
    lamb_hat_list = []

    a_err = []
    B_err = []
    lamb_err = []

    # we should get trials many results from QUEUE_OUT
    for _ in tqdm(range(NUM_TRIALS)):
        tid, a_hat, B_hat, lamb_hat = QUEUE_OUT.get(block=True)

        trial_ids_out.append(tid)

        a_hat_list.append(a_hat)
        B_hat_list.append(B_hat)
        lamb_hat_list.append(lamb_hat)

        # compute error
        a_err.append(np.linalg.norm(a_hat - a_list[tid]))
        B_err.append(compute_B_err(B_hat, B_list[tid]))
        lamb_err.append(abs(lamb_hat - lamb_list[tid]))

    a_err = np.array(a_err)
    B_err = np.array(B_err)
    lamb_err = np.array(lamb_err)

    # save each trial result
    for i in range(NUM_TRIALS):
        write_fpme_trial(
            ng,
            nc,
            trial_ids_out[i],
            a_hat_list[i],
            B_hat_list[i],
            lamb_hat_list[i],
        )

    # save the trial summary
    write_fpme_trial_summary(ng, nc, a_err, B_err, lamb_err)


if __name__ == "__main__":
    # start the procs
    procs = []
    for i in range(NUM_PROCS):
        proc = mp.Process(target=proc_run_trials, args=(i,))
        proc.start()
        procs.append(proc)

    for ng in range(2, 6):
        for nc in range(2, 6):
            print("m={} k={}".format(ng, nc))
            q = nc ** 2 - nc
            manage_trials(ng, nc, q)

    QUEUE_IN.put(None)  # signal end to procs
