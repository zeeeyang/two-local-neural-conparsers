CMAKE_MINIMUM_REQUIRED(VERSION 2.8) 
foreach(TARGET BILSTM-CKY-PARSER)
  ADD_EXECUTABLE(${TARGET} ${TARGET}.cc)
  if (WITH_CUDA_BACKEND)
    target_link_libraries(${TARGET} gdynet ${LIBS})
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  else()
    target_link_libraries(${TARGET} dynet ${LIBS})
  endif (WITH_CUDA_BACKEND)
  if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} rt)
  endif()
endforeach()

