TO=10
RD=log/45-17-7
mkdir -p $RD
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc cadical103 incremental 2>&1 | tee $RD/cadical103.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc cadical153 incremental 2>&1 | tee $RD/cadical153.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc cadical195 incremental 2>&1 | tee $RD/cadical195.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc cryptominisat incremental 2>&1 | tee $RD/cryptominisat.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc gluecard3 incremental 2>&1 | tee $RD/gluecard3.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc gluecard4 incremental 2>&1 | tee $RD/gluecard4.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc glucose3 incremental 2>&1 | tee $RD/glucose3.log

runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc glucose4 incremental 2>&1 | tee $RD/glucose4.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc glucose42 incremental 2>&1 | tee $RD/glucose42.log

runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc lingeling incremental 2>&1 | tee $RD/lingeling.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc maplechrono incremental 2>&1 | tee $RD/maplechrono.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc maplecm incremental 2>&1 | tee $RD/maplecm.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc maplesat incremental 2>&1 | tee $RD/maplesat.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc mergesat3 incremental 2>&1 | tee $RD/mergesat3.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc minicard incremental 2>&1 | tee $RD/minicard.log
runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc minisat22 incremental 2>&1 | tee $RD/minisat22.log

runlim -r $TO python3 -u two_solver.py 45-17 7 nsc nsc minisatgh incremental 2>&1 | tee $RD/minisatgh.log
