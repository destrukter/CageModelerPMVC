set(MCD_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")

configure_file("Configuration.h.in" "${MCD_GENERATED_INCLUDE_DIR}/Configuration.h" ESCAPE_QUOTES)
include_directories("${MCD_GENERATED_INCLUDE_DIR}")