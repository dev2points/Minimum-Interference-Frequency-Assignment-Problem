TO=60
RD=log/45-17-7
mkdir -p $RD
runlim -r $TO python3 -u test.py 45-17 7 nsc tot cadical103 assumptions 2>&1 | tee $RD/cadical103.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot cadical153 assumptions 2>&1 | tee $RD/cadical153.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot cadical195 assumptions 2>&1 | tee $RD/cadical195.log

runlim -r $TO python3 -u test.py 45-17 7 nsc tot gluecard3 assumptions 2>&1 | tee $RD/gluecard3.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot gluecard4 assumptions 2>&1 | tee $RD/gluecard4.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot glucose3 assumptions 2>&1 | tee $RD/glucose3.log

runlim -r $TO python3 -u test.py 45-17 7 nsc tot glucose4 assumptions 2>&1 | tee $RD/glucose4.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot glucose42 assumptions 2>&1 | tee $RD/glucose42.log

runlim -r $TO python3 -u test.py 45-17 7 nsc tot lingeling assumptions 2>&1 | tee $RD/lingeling.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot maplechrono assumptions 2>&1 | tee $RD/maplechrono.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot maplecm assumptions 2>&1 | tee $RD/maplecm.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot maplesat assumptions 2>&1 | tee $RD/maplesat.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot mergesat3 assumptions 2>&1 | tee $RD/mergesat3.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot minicard assumptions 2>&1 | tee $RD/minicard.log
runlim -r $TO python3 -u test.py 45-17 7 nsc tot minisat22 assumptions 2>&1 | tee $RD/minisat22.log




