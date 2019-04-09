# surrogate_model_search
An experimental comparison between GA search guided by simulation and GA search guided by surrogate models trained to predict fitness function component values.

Within the context of a population dynamics agent-based model we search for the emergent behaviour of oscillating populations of predator and prey.

Simulation responses consist of 3 observations determining if the current simulation initial conditins have resulted in the display of oscillating populations.

ANN surrogate models are trained on simulation data generated in a HPC environment with 4 GPU devices avialable for batched parallel model simulation.

Folders are created programmatically by "surrogate_modelling_search_experiment.py" which needs no arguments to be provided. To adjust experimental parameters, edit variables at the start of the aforementioned file.

Graph generation is handled by "plot_smse_graphs.py"