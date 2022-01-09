# Artifact Overview of Tzer (OOPSLA'22)

## Get Started

### Test-bed Requirement

- **OS:** A Linux System with [Docker](https://docs.docker.com/get-docker/) Support;
- **Hardware**
  - X86 CPU; 8GB RAM; 256GB Storage; Good Network to [GitHub](https://github.com/) and [Docker Hub](https://hub.docker.com/);

### Quick Start with Docker

```shell
# Pull docker image from docker hub;
docker run --rm -it tzerbot/oopsla

# Inside the image; 
cd tzer
python3 src/main_tir.py   --fuzz-time 10     --report-folder ten-minute-fuzz
#                         run for 10 min.    bugs in folder `ten-minute-fuzz`
```

After a successful installation, commands above run for 10 minutes and print something like:

![](./docs/imgs/tzer-terminal-output.png)

Output files are stored in `ten-minute-fuzz` (parameter of `--report-folder`):

<details><summary><b>Report folder contents</b> <i>[click to expand]</i></summary>
<div>

- `cov_by_time.txt`: a csv file where columns means "time" (second) and edge coverage;
- `${BUG_TYPE}_${BUG_ID}.error_message.txt`: error message snapshot of failures;
- `${BUG_TYPE}_${BUG_ID}.ctx`: context data to reproduce bugs (stored in Pickle. See [config.py](src/tzer/context.py#L51))
- `meta.txt`: metadata including git version of TVM and experiment time;
- `tir_by_time.pickle`: generated <F, P> (i.e., TIR and Passes) files (if `TIR_REC=1` is set);
- `valid_seed_new_cov_count.txt`: number of generated valid tests with new coverage;

</div>
</details>

<details><summary><b>Main commandline options</b> <i>[click to expand]</i></summary>
<div>

Commandline options (added as tail of commands):

- `--fuzz-time`: Time budget of fuzzing (minute);
- `--tolerance`: Parameter $N_{max}$ in the paper (control the interleaving of IR and pass mutation);
- `--report-folder`: Path to store results (e.g., coverage trend);

Environment variables to control the algorithm options (added the prefix of commands):

- `PASS=1` to enable pass mutation;
- `NO_SEEDS=1` to disable initial seeds (start from an empty function);
- `NO_COV=1` to disable the coverage feedback;
- `TIR_REC=1`to record generated TIR files (for evaluating non-coverage version);

</div>
</details>


## Step by Step Instructions

### Claim 1: Bug Finding (25 minutes)

> (Abstract) "To date, Tzer has detected **40** previously unknown bugs for TVM, with **30** bugs confirmed and **24** bugs fixed (PR merged)."

#### The total 40 bugs

Reproducing these bugs requires GPU environments and retrieving to the initial git commit when we started our bug finding. 
To ease the effort of reviewers, we set up a cloud environment in Colab to reproduce the bugs in your browser remotely.
To reproduce the bugs, open the Colab [link](https://colab.research.google.com/github/Tzer-AnonBot/tzer/blob/main/bug-report.ipynb) in your browser and follow the instructions to set up the GPU environment and reproduce the bugs with a simple click on your mouse. :-)

#### 30 confirmed and 24 fixed

We compiled the confirmed bug under the Google Sheet [link](https://docs.google.com/spreadsheets/d/1CFHUBtCtuPOrGw7W-GLLXdpV3wdJHUDdP43UvKkAE4Q/edit?usp=sharing).
Trackable links to PR/Bug reports are listed in the Google Sheet.
Note that more than 1 bugs might be included for one PR/Issue link.


### Claim 2: RQ1 - Comparison with Existing Work (26 hours)

We list steps to reproduce results in Section 5.1.

*Note that there will be randomness in fuzzing given different system performance and random seeds.*
*The detailed numbers might not be strictly reproducible but the overall trend should be consistent.*

#### Figure 5 and Table 2

Figure 5 and Table 2 can be reproduced under the same batch of experiments.
For each experiment, the output is the the report folder. 
Under the report folder, `cov_by_time.txt`: for each it represents elapsed time (in seconds) and coverage splited by ",".
In `valid_seed_new_cov_count.txt`, the number of lines represents the total number of valuable tests (initial seeds not taken into account).

**Tzer**

- report folder: `/tzer/tzer-tzer-seed`
- expected time: 4 hours
- to reproduce:

```shell
# In the container
cd /tzer
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tzer-tzer-seed --tolerance 4
```

**Tzer (LEMON seeds)**

- report folder: `/tzer/tzer-lemon-seed`
- expected time: 4 hours
- to reproduce:

```shell
# In the container
cd /tzer
TBD@Sen
```

**LEMON**

- report folder: `/tzer/lemon`
- expected time: 1 hour
- to reproduce: TBD

Directly using [LEMON](https://github.com/Jacob-yen/LEMON) to generate DL models from scratch is complicated and the massive generated models consumes disk spaces at TB level. It requires using LEMON's docker image and needs some edits to make the pipeline work. LEMON generates Keras models (hundreds of Gigabytes and even over 1 Terabyte). To evaluate it under TVM, we also need to covert these Keras models into TIR files. 

To ease the reproduction procedure, we directly provide the minimized valuable TIR files converted from LEMON:

```shell
# In the container
cd /tzer
TBD@Sen
```

**LibFuzzer**

- report folder: `/tzer/libfuzz`
- expected time: 4 hours
- to reproduce:

```shell
# In the container
cd /tzer
TBD@Sen
```

**TVMFuzz**

- report folder: `/tzer/tvm-fuzz`
- expected time: 10 hours
- to reproduce:

```shell
# In the container
cd /tzer
TBD@Sen
```

**Result Visualization**

To visualize the results:

```shell
cd /tzer
python3 src/plot_cov.py -f tzer-tzer-seed tzer-lemon-seed lemon libfuzz tvm-fuzz -cl 5000
```

Check `/tzer/cov.png`.



### Claim 3: RQ2 - Ablation Study (20 hours)

We list steps to reproduce results in Section 5.2.

*Note that there will be randomness in fuzzing given different system performance and random seeds.*
*The detailed numbers might not be strictly reproducible but the overall trend should be consistent.*

#### Figure 6: Ablation Study

**(1): General IR Mut. (No Cov.)**

- report folder: `/tzer/ablation-1`
- expected time: 8 hours
- to reproduce:
 
```shell
# in the docker image
cd /tzer
TVM_HOME=$TVM_NO_COV_HOME PYTHONPATH=$TVM_HOME/python TIR_REC=1 NO_COV=1 python3 src/main_tir.py --fuzz-time 240 --report-folder ablation-1
python3 src/get_cov.py --folders ablation-1 # Evaluate samples on instrumented TVM to get coverage results.
```

**(2): (1) + Cov. Guidance**

- report folder: `/tzer/ablation-2`
- expected time: 4 hours
- to reproduce:

```shell
# in the docker image
cd /tzer
python3 src/main_tir.py --fuzz-time 240 --report-folder ablation-2
```

**(3): (2) + Domain-Specific IR Mutation**

- report folder: `/tzer/ablation-3`
- expected time: 4 hours
- to reproduce:

```shell
# in the docker image
cd /tzer
LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder ablation-3
```

**(4): (3) + Random Pass Mutation**

- report folder: `/tzer/ablation-4`
- expected time: 4 hours
- to reproduce:

```shell
# in the docker image
cd /tzer
PASS=1 RANDOM_PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder ablation-4
```

**(5): (3) + Evolutionary IR-Pass Mutation**

- This experiment can reuse `/tzer/tzer-tzer-seed` from experiments for Figure 5.
- report folder: `/tzer/ablation-5` <- `/tzer/tzer-tzer-seed`


```shell
# in the docker image
cp -r /tzer/tzer-tzer-seed /tzer/ablation-5
```

**Result Visualization**

To visualize the results:

```shell
cd /tzer
python3 src/plot_cov.py -f ablation-1 ablation-2 ablation-3 ablation-4 ablation-5 -cl 18000
```

Check `/tzer/cov.png`.



### Claim 4: RQ3 - Parameter Sensitivity (44 hours)

We list steps to reproduce results in Section 5.3.

*Note that there will be randomness in fuzzing given different system performance and random seeds.*
*The detailed numbers might not be strictly reproducible but the overall trend should be consistent.*

#### Figure 7 (4 hours)

**Tzer without seeds**

- report folder: `/tzer/tzer-without-seed`
- expected time: 4 hours
- to reproduce:

```shell
# In the container
cd /tzer
NO_SEEDS=1 PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tzer-without-seed --tolerance 4
```

**Tzer with seed**

- This experiment can reuse `/tzer/tzer-tzer-seed` from experiments for Figure 5.
- report folder: `/tzer/tzer-with-seed` <- `/tzer/tzer-tzer-seed`


```shell
# in the docker image
cp -r /tzer/tzer-tzer-seed /tzer/tzer-with-seed
```

**Result Visualization**

To visualize the results:

```shell
cd /tzer
python3 src/plot_cov.py -f tzer-with-seed tzer-without-seed -cl 20000
```

Check `/tzer/cov.png`.

#### Figure 8 and 9 (40 hours)

```shell
cd /tzer
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tolerance-1 --tolerance 1
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tolerance-2 --tolerance 2
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tolerance-3 --tolerance 3
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tolerance-4 --tolerance 4
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tolerance-5 --tolerance 5
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tolerance-6 --tolerance 6
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tolerance-7 --tolerance 7
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tolerance-8 --tolerance 8
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tolerance-9 --tolerance 9
PASS=1 LOW=1 python3 src/main_tir.py --fuzz-time 240 --report-folder tolerance-10 --tolerance 10
```

**Result Visualization**

To visualize the results:

```shell
cd /tzer
python3 src/plot_cov.py -f ./tolerance-* -cl 23000
```

Check `/tzer/cov.png`.



### Claim 5: RQ4 - Bug Detection Effectiveness

We list steps to reproduce results in Section 5.4.

#### Table 4

The classification and detectable bugs is summarised in this [link](https://docs.google.com/spreadsheets/d/1CFHUBtCtuPOrGw7W-GLLXdpV3wdJHUDdP43UvKkAE4Q/edit?usp=sharing).
Columns 5-7 from the [Google Sheet](https://docs.google.com/spreadsheets/d/1CFHUBtCtuPOrGw7W-GLLXdpV3wdJHUDdP43UvKkAE4Q/edit?usp=sharing) specifies detectable bugs by other baselines among all bugs detected by Tzer which justifies column 2-4 in Table 4 of our paper.
Column 8 from the [Google Sheet](https://docs.google.com/spreadsheets/d/1CFHUBtCtuPOrGw7W-GLLXdpV3wdJHUDdP43UvKkAE4Q/edit?usp=sharing) indicates the bugs that do not have to be triggered by combinations of passes and IRs.
