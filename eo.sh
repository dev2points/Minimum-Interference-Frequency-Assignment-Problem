TO=600
RD=log/eo
mkdir -p $RD

# runlim -r $TO python3 -u test.py 45-17 7 nsc tot cadical103 assumptions 2>&1 | tee $RD/nsc.log
runlim -r $TO python3 -u test.py 45-17 7 sc tot cadical103 assumptions 2>&1 | tee $RD/sc.log
runlim -r $TO python3 -u test.py 45-17 7 pairwise tot cadical103 assumptions 2>&1 | tee $RD/pairwise.log

runlim -r $TO python3 -u test.py 45-17 7 card_0 tot cadical103 assumptions 2>&1 | tee $RD/card_0.log
runlim -r $TO python3 -u test.py 45-17 7 card_1 tot cadical103 assumptions 2>&1 | tee $RD/card_1.log
runlim -r $TO python3 -u test.py 45-17 7 card_2 tot cadical103 assumptions 2>&1 | tee $RD/card_2.log
runlim -r $TO python3 -u test.py 45-17 7 card_3 tot cadical103 assumptions 2>&1 | tee $RD/card_3.log
runlim -r $TO python3 -u test.py 45-17 7 card_4 tot cadical103 assumptions 2>&1 | tee $RD/card_4.log
runlim -r $TO python3 -u test.py 45-17 7 card_5 tot cadical103 assumptions 2>&1 | tee $RD/card_5.log
runlim -r $TO python3 -u test.py 45-17 7 card_6 tot cadical103 assumptions 2>&1 | tee $RD/card_6.log
runlim -r $TO python3 -u test.py 45-17 7 card_7 tot cadical103 assumptions 2>&1 | tee $RD/card_7.log
runlim -r $TO python3 -u test.py 45-17 7 card_8 tot cadical103 assumptions 2>&1 | tee $RD/card_8.log
runlim -r $TO python3 -u test.py 45-17 7 card_9 tot cadical103 assumptions 2>&1 | tee $RD/card_9.log

runlim -r $TO python3 -u test.py 45-17 7 pb_0 tot cadical103 assumptions 2>&1 | tee $RD/pb_0.log
runlim -r $TO python3 -u test.py 45-17 7 pb_1 tot cadical103 assumptions 2>&1 | tee $RD/pb_1.log
runlim -r $TO python3 -u test.py 45-17 7 pb_2 tot cadical103 assumptions 2>&1 | tee $RD/pb_2.log
runlim -r $TO python3 -u test.py 45-17 7 pb_3 tot cadical103 assumptions 2>&1 | tee $RD/pb_3.log
runlim -r $TO python3 -u test.py 45-17 7 pb_4 tot cadical103 assumptions 2>&1 | tee $RD/pb_4.log
runlim -r $TO python3 -u test.py 45-17 7 pb_5 tot cadical103 assumptions 2>&1 | tee $RD/pb_5.log
runlim -r $TO python3 -u test.py 45-17 7 pb_6 tot cadical103 assumptions 2>&1 | tee $RD/pb_6.log
runlim -r $TO python3 -u test.py 45-17 7 pb_7 tot cadical103 assumptions 2>&1 | tee $RD/pb_7.log
runlim -r $TO python3 -u test.py 45-17 7 pb_8 tot cadical103 assumptions 2>&1 | tee $RD/pb_8.log
runlim -r $TO python3 -u test.py 45-17 7 pb_9 tot cadical103 assumptions 2>&1 | tee $RD/pb_9.log