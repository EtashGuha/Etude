#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "MLi4.36" for configuration "Release"
set_property(TARGET MLi4.36 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MLi4.36 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/CompilerAdditions/libMLi4.36.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS MLi4.36 )
list(APPEND _IMPORT_CHECK_FILES_FOR_MLi4.36 "${_IMPORT_PREFIX}/CompilerAdditions/libMLi4.36.a" )

# Import target "mathlink" for configuration "Release"
set_property(TARGET mathlink APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mathlink PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/CompilerAdditions/mathlink.framework/Versions/4.36/mathlink"
  IMPORTED_SONAME_RELEASE "@executable_path/../Frameworks/mathlink.framework/Versions/4.36/mathlink"
  )

list(APPEND _IMPORT_CHECK_TARGETS mathlink )
list(APPEND _IMPORT_CHECK_FILES_FOR_mathlink "${_IMPORT_PREFIX}/CompilerAdditions/mathlink.framework/Versions/4.36/mathlink" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
