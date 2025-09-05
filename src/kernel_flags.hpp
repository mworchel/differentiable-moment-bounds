#pragma once

#include <cstdint>

#include "common.hpp"

// Flags for the `ComputeMomentBounds...` kernels
// NOTE: These flags are tightly coupled to the code in `kernels.cu`, 
// specifically the macros that instantiate the kernels.
enum class ComputeMomentBoundsFlags : uint32_t
{
	None = 0b00,
	RetainRoots = 0b01,
	RetainWeights = 0b10, // Only valid if RetainRoots is also set
	All = 0b11,
};

DEVICE constexpr uint32_t operator+(ComputeMomentBoundsFlags flag)
{
	return static_cast<uint32_t>(flag);
}