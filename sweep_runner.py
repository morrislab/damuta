import wandb
import os
import damut
​
if __name__ == "__main__":
​
	defaults_fp = "config/config-defaults.yml"

	# init run
	run = wandb.init()
	# configure with defaults and sweep args
	conf = damut.load_sweep_config(run, defaults_fp)

	# fit model
	damut.infer(conf)