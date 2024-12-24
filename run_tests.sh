#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

RESUME="Data/Resumes/A_Abdo_Resume_July_2024.pdf"
JD_DIR="Data/JobDescription"
DIVIDER="=============================================="

for jd in ${JD_DIR}/job_desc_*.pdf; do
    echo -e "\n${YELLOW}${DIVIDER}${NC}"
    echo -e "${BOLD}${GREEN}🔍 Analyzing Resume Match${NC}"
    echo -e "${BLUE}📄 Resume:${NC} $(basename $RESUME)"
    echo -e "${BLUE}📋 Job Description:${NC} $(basename $jd)"
    echo -e "${YELLOW}${DIVIDER}${NC}\n"
    
    CMD="time python match_resume_jd_optimized.py \"$RESUME\" \"$jd\""
    echo -e "${BOLD}⏳ Running:${NC} $CMD\n"
    time eval "$CMD"
    
    echo -e "\n${YELLOW}${DIVIDER}${NC}\n"
done