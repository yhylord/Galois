if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/slabhash")
  message(FATAL_ERROR "Cannot find the slabhash Git submodule. Please run `git submodule update --init --recursive`")
endif()

target_include_directories(galois_gpu PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/slabhash/src>
  $<INSTALL_INTERFACE:include/galois/gpu/slabhash/src>
)
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/external/slabhash
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/galois/gpu/slabhash"
  COMPONENT dev
)

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/slabhash/SlabAlloc")
  message(FATAL_ERROR "Cannot find the SlabAlloc Git submodule. Please run `git submodule update --init --recursive`")
endif()

target_include_directories(galois_gpu PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/slabhash/SlabAlloc/src>
  $<INSTALL_INTERFACE:include/galois/gpu/SlabAlloc/src>
)
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/external/slabhash/SlabAlloc
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/galois/gpu/SlabAlloc"
  COMPONENT dev
)

app_dist(bc_level betweennesscentrality-level)
add_test_dist(betweennesscentrality-level-dist rmat15 NO_ASYNC ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -numOfSources=4)

app_dist(bc_mr betweennesscentrality-minrounds)
add_test_dist(betweennesscentrality-minrounds-dist rmat15 NO_ASYNC ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -numOfSources=4 -numRoundSources=4)
add_test_dist(betweennesscentrality-minrounds-dist rmat15all NO_ASYNC NOT_QUICK ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -numRoundSources=4096)
