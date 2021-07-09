# Quadratic Metric Elicitation

This repository contains the source code for the paper *Quadratic Metric Elicitation with Application to Fairness*, submitted to AI STATS 2021.

An overview of the files:
1. Source code for Linear Performance Metric Elicitation (LPME), Quadratic Performance Metric Elicitation (QPME), and Fair Performance Metric Elicitation (FPME) algorithms.
	- qme/common.py
	- qme/lpme.py
	- qme/qme.py
	- qme/fpme.py
	- qme/fpme_utils.py
	<br />

2. Code for running trials to see how the algorithms perform on different inputs.
	- qme/ipynb/trials.py
	- qme/ipynb/qme.ipynb
	- qme/ipynb/fpme.ipynb
	- qme/ipynb/fpme_trials_runner.py
	<br />

3. Code for visualizing results and demonstrating performance against baselines.
	- qme/ipynb/qme_results_analyze.ipynb
	- qme/ipynb/fpme_results_analyze.ipynb
	- qme/ipynb/data/processing.ipynb
	- qme/ipynb/data/classifiers_wo_constraints.ipynb
	- qme/ipynb/ranker_analyze.ipynb
	<br />

4. Code for investigating and understanding the algorithms as well as issues they may run into in practice.
	- qme/ipynb/qme_theoretical.ipynb
	- qme/ipynb/fpme_theoretical.ipynb
	- qme/ipynb/qme.ipynb
	- qme/ipynb/fpme.ipynb
	- qme/ipynb/explore_fractional_error.ipynb
	<br />

To get started, first install dependencies:
```
pip install requirements.txt
```

Then, make sure everything works by running the examples in *qme/ipynb/lpme.ipynb*, *qme/ipynb/qme.ipynb*, and *qme/ipynb/fpme.ipynb*. This should give you a feel for the code and how it runs. Then, we recommend either diving into the source or looking at the *_theoretical.ipynb* files.

To reproduce the QME and FPME figures in the paper, run:
```
cd qme/ipynb
python trials.py
```

Then, interactively run *qme/ipynb/qme.ipynb* and *qme/ipynb/fpme.ipynb*. The former can finish in 10-20 minutes, whereas the latter can take over an hour. *qme/ipynb/fpme_trials_runner.py* exists to run the latter continuously. **THESE FILES USE MULTIPROCESSING TO REDUCE RUNTIME AND WILL SIGNIFICANTLY SLOW DOWN YOUR COMPUTER**. Modify the source to accomodate your system.

To reproduce the ranking graphs in the paper, run *qme/ipynb/ranker_analyze.ipynb*.
