file(REMOVE_RECURSE
  "../fiber.pdb"
  "../fiber"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/fiber.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
