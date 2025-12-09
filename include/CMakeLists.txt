file(GLOB_RECURSE _headers ${CMAKE_CURRENT_LIST_DIR}/*.h  CONFIGURE_DEPEND)
file(GLOB_RECURSE _private_headers ${CMAKE_CURRENT_LIST_DIR}/*_p.h CONFIGURE_DEPEND)
list(REMOVE_ITEM _headers ${_private_headers})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/splat/splat_version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/splat/splat_version.h
)

generate_export_header(
	splat
	EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/splat/exports.h
	EXPORT_MACRO_NAME SPLAT_EXPORT
)

set(_headers ${_headers} 
	${CMAKE_CURRENT_BINARY_DIR}/splat/exports.h
	${CMAKE_CURRENT_BINARY_DIR}/splat/splat_version.h
)

target_sources(splat PUBLIC ${_headers})
target_sources(splat PRIVATE ${_private_headers})
unset(_headers)
unset(_private_headers)

target_include_directories(splat
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include/splat>)