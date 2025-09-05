#include "log.hpp"

#include <cstdarg>
#include <cstdio>
#include <iterator>

char const* format_message(char const* format, ...)
{
    thread_local char buffer[1024] = { 0 };

    std::va_list args;
    va_start(args, format);
    int count = vsnprintf(buffer, std::size(buffer), format, args);
    va_end(args);

    if (count < 0 || count >= static_cast<int>(std::size(buffer))) {
        // Handle encoding error/truncation
        buffer[std::size(buffer) - 1] = '\0';
    }

    return buffer;
}