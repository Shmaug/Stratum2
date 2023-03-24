// gpu-printing.h
#pragma once

// This file provides the CPU side support for a basic GPU
// printing system. The GPU implementation of the system
// is in `printing.slang`.

// The host side of the system needs to be able to load
// strings that were specified in Slang shader code, and
// for that it will use the Slang reflection API.
//
#include <slang/slang.h>

// We also need a way to store the data for strings that
// were used in shader code, and we will go ahead and
// use the C++ STL for that, in order to make this
// code moderately portable.
//
#include <map>
#include <string>

    /// Stores state used for executing print commands generated by GPU shaders
struct GPUPrinting
{
public:
        /// Load any string literals used by a Slang program.
        ///
        /// The `slangReflection` should be the layout and reflection
        /// object for a Slang shader program that might need to produce
        /// printed output. This function will load any strings
        /// referenced by the program into its database for mapping
        /// string hashes back to the original strings.
        ///
    void loadStrings(slang::ProgramLayout* slangReflection);

        /// Process a buffer of GPU printing commands and write output to `stdout`.
        ///
        /// This function attempts to read print commands from the buffer
        /// pointed to by `data` and execute them to produce output.
        ///
        /// The buffer pointed at by `data` (of size `dataSize`) should be allocated
        /// in host-visible memory.
        ///
        /// Before executing GPU work, the first four bytes pointed to by `data`
        /// should have been cleared to zero.
        ///
        /// If GPU work has attempted to write more data than the buffer
        /// can fit, a warning will be printed to `stderr`, and printing commands
        /// that could not fit completely in the buffer will be skipped.
        ///
    void processGPUPrintCommands(const void* data, size_t dataSize);

private:
    typedef int StringHash;

    std::map<StringHash, std::string> m_hashedStrings;
};
