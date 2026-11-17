#!/bin/bash
cd /home/bruno/Documents/ml4ce_2026-main/BO_coursework/dev
sleep 5; while pgrep -f "q_w2.sh" >/dev/null; do sleep 20; done
bash logs/register/run_register.sh > logs/register/register.log 2>&1
../.venv/bin/python make_register.py > logs/register/make_register.out 2>&1
date >> logs/register/register.log
