# Source Separation Code
In order to run the code, you have to go to the following folder: `UNet/separation_model/training`.

Then, you have to run, as an example, the following command:

`python train_unet.py --loss-type speccosloss --imag-penalty 10000 --schedule --nb-resblocks 2 --start-fmaps 44 --batch-size 24 --dropcopy 0.1 --nb-copies 10 --avg-copies --experiment-path ../experiments/runtest >> ../experiments/runtest/results_file.out 2>&1`

The results would be then written in real time in `UNet/separation_model/experiments/runtest/results_file.out `