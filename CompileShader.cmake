function(stm_compile_shader SRC_PATH DST_FOLDER)
	cmake_parse_arguments(PARSED "" "" "INCLUDES" ${ARGN})

	get_filename_component(SRC_NAME ${SRC_PATH} NAME_WLE)
	file(STRINGS ${SRC_PATH} LINES)

	set(DST_TARGETS "")
	set(DST_FILES "")

	# Scan the file for any lines beginning with '#pragma compile'
	foreach(LINE IN LISTS LINES)
		string(FIND "${LINE}" "#pragma compile" R)
		if(NOT ${R} EQUAL 0)
		continue()
		endif()

		set(COMPILE_CMD "")
		separate_arguments(COMPILE_CMD UNIX_COMMAND ${LINE})
		list(REMOVE_AT COMPILE_CMD 0 1) # skip '#pragma compile'

		# detect entry point

		set(ENTRY_POINT "")
		if ("${COMPILE_CMD}" MATCHES "-fentry-point=([A-Za-z0-9_]+)")
			set(ENTRY_POINT ${CMAKE_MATCH_1})
		else()
			list(FIND COMPILE_CMD "-E" ENTRY_POINT)
			if (ENTRY_POINT EQUAL -1)
				list(FIND COMPILE_CMD "-entry" ENTRY_POINT)
			endif()

			if (NOT ENTRY_POINT EQUAL -1)
				math(EXPR ENTRY_POINT "${ENTRY_POINT}+1")
				list(GET COMPILE_CMD ${ENTRY_POINT} ENTRY_POINT)
			else()
				message(WARNING "${SRC_PATH}: '#pragma compile' with no entry point")
				continue()
			endif()
		endif()

		if (${ENTRY_POINT} STREQUAL "main")
			set(DST_NAME "${SRC_NAME}")
		else()
			set(DST_NAME "${SRC_NAME}_${ENTRY_POINT}")
		endif()

		set(SPV_PATH "${DST_FOLDER}/${DST_NAME}.spv")

		# add includes
		foreach (INC_PATH ${PARSED_INCLUDES})
			list(APPEND COMPILE_CMD "-I" "${INC_PATH}")
		endforeach()

		# use internal slangc
		list(POP_FRONT COMPILE_CMD)
		list(PREPEND COMPILE_CMD "${SLANGC}")
		list(APPEND COMPILE_CMD "-target" "spirv")
		list(APPEND COMPILE_CMD "-D__HLSL__")
		list(APPEND COMPILE_CMD "-D__SLANG__")
		list(APPEND COMPILE_CMD "-o" "${SPV_PATH}")

		# add commands

		add_custom_command(OUTPUT "${SPV_PATH}"
			COMMAND ${COMPILE_CMD} ${SRC_PATH}
			DEPENDS "${SRC_PATH}" IMPLICIT_DEPENDS CXX "${SRC_PATH}")

		add_custom_target(${DST_NAME} ALL DEPENDS "${SPV_PATH}")

		list(APPEND DST_TARGETS ${DST_NAME})
		list(APPEND DST_FILES "${SPV_PATH}")
	endforeach()

	set(DST_TARGETS ${DST_TARGETS} PARENT_SCOPE)
	set(DST_FILES ${DST_FILES} PARENT_SCOPE)
endfunction()

function(stm_add_shaders)
	cmake_parse_arguments(PARSED "" "DST_FOLDER" "SOURCES;DEPENDS;INCLUDES" ${ARGN})
	foreach(SHADER_FILE ${PARSED_SOURCES})
		stm_compile_shader("${SHADER_FILE}" "${PARSED_DST_FOLDER}" INCLUDES ${PARSED_INCLUDES})

		if (DST_TARGETS)
			foreach(DEP ${PARSED_DEPENDS})
				add_dependencies(${DEP} ${DST_TARGETS})
			endforeach()
		endif()

		foreach(DST_FILE ${DST_FILES})
			install(FILES ${DST_FILE} DESTINATION bin/Shaders)
		endforeach()
	endforeach()
endfunction()