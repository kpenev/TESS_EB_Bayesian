tic=$1 

systemd-run --scope -p MemoryMax=24G --user \
    ./visualize.py $tic \
    --plot-convergence tess${tic}_convergence.pdf \
    --chain-expression '$M_1+M_2$=mtotal' \
    --chain-expression '$M_2/M_1$=mratio' \
    --chain-expression 'Age=age_gyr' \
    --chain-expression '$[M/H]$=meh' \
    --chain-expression '$P_{orb}$=per' \
    --chain-expression 'e=ecc' \
    --chain-expression '$\omega$=w' \
    --chain-expression 'b=primary_impact_param' \
    --chain-expression '$T_0$=eclipse_time' \
    > \
    plot_convergence.out 2>&1 &
