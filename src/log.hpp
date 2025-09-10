#pragma once

// Returns a formatted string valid until the next call to format_message() in the same thread. Intended for immediate use only.
char const* format_message(char const* format, ...);