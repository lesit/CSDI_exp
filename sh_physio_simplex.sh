echo exe_physio missing - $1
nohup python exe_physio.py --testmissingratio $1 --nsample 100 --noise_fn simplex > /dev/null 2>&1 &
