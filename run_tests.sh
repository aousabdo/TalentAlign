#!/bin/bash

RESUME="Data/Resumes/A_Abdo_Resume_July_2024.pdf"
JD_DIR="Data/JobDescription"

for jd in ${JD_DIR}/job_desc_*.pdf; do
    echo "----------------------------------------"
    echo "Testing $(basename $RESUME) against $(basename $jd)"
    time python match_resume_jd_optimized.py "$RESUME" "$jd"
    echo "----------------------------------------"
done