#!/usr/bin/env bash

export DATA_DIR="data/msd/"

mkdir -p ${DATA_DIR}/train/
mkdir -p ${DATA_DIR}/val/
mkdir -p ${DATA_DIR}/val_src/
mkdir -p ${DATA_DIR}/test/
mkdir -p ${DATA_DIR}/test_src/
mkdir -p ${DATA_DIR}/full_train/

cat <(printf 'user,song,listens\n') \
  <(sed 's/\t/,/g' train_triplets.txt) \
  > ${DATA_DIR}train/train.csv

cat <(printf 'user,song,listens\n') \
  <(sed 's/\t/,/g' EvalDataYear1MSDWebsite/year1_valid_triplets_hidden.txt) \
  > ${DATA_DIR}val/val.csv

cat <(printf 'user,song,listens\n') \
  <(sed 's/\t/,/g' EvalDataYear1MSDWebsite/year1_valid_triplets_visible.txt) \
  > ${DATA_DIR}val_src/val.csv

cat <(printf 'user,song,listens\n') \
  <(sed 's/\t/,/g' EvalDataYear1MSDWebsite/year1_test_triplets_hidden.txt) \
  > ${DATA_DIR}test/test.csv

cat <(printf 'user,song,listens\n') \
  <(sed 's/\t/,/g' EvalDataYear1MSDWebsite/year1_test_triplets_visible.txt) \
  > ${DATA_DIR}test_src/test.csv

cat <(printf 'user,song,listens\n') <(sed 's/\t/,/g' ${DATA_DIR}train_triplets.txt) \
  <(sed 's/\t/,/g' ${DATA_DIR}EvalDataYear1MSDWebsite/year1_valid_triplets_visible.txt) \
  <(sed 's/\t/,/g' ${DATA_DIR}EvalDataYear1MSDWebsite/year1_test_triplets_visible.txt) \
  > ${DATA_DIR}full_train/full_train.csv

