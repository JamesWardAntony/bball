#! /bin/bash

module load pyger 

set -e #stop immediately when error occurs

subj=$1

jupyter nbconvert \
  --ExecutePreprocessor.allow_errors=True \
  --ExecutePreprocessor.timeout=-1 \
  --FilesWriter.build_directory=./notebook-01-output \
  --execute 01-ImpConcatZ-ses01_sub-$subj.ipynb
