app_dist(bc_level betweennesscentrality-level)
add_test_dist(betweennesscentrality-level-dist rmat15 NO_ASYNC ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -numOfSources=4)

app_dist(bc_mr betweennesscentrality-minrounds)
add_test_dist(betweennesscentrality-minrounds-dist rmat15 NO_ASYNC ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -numOfSources=4 -numRoundSources=4)
add_test_dist(betweennesscentrality-minrounds-dist rmat15all NO_ASYNC NOT_QUICK ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -numRoundSources=4096)
