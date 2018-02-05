

floyd run --data amoussawi/projects/floydhub-test/8/output:/data \
  --data amoussawi/projects/floydhub-test/19/output:/model \
  --env pytorch-0.2 "export PYTHONPATH=$PYTHONPATH:./ ; python jobs/msd_train.py"

