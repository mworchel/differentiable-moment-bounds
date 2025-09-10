#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Expected 3 arguments (input path, output path, name)" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream file_in(argv[1]);
    if (!file_in.is_open())
    {
        std::cerr << "Unable to load file '" << argv[1] << "'." << std::endl;
        return EXIT_FAILURE;
    }

    file_in.seekg(0, std::ios::end);
    size_t size = file_in.tellg();
    std::string input_content(size, ' ');
    file_in.seekg(0);
    file_in.read(&input_content[0], size);
    file_in.close();

    std::stringstream stream;
    stream << "char const " << argv[3] << "[" << input_content.size() + 1 << "] = {" << std::endl;
    for (std::size_t i = 0; i < input_content.size(); ++i)
    {
        stream << std::hex << "0x" << (0xFF & input_content[i]) << ", ";
        if ((i + 1) % 20 == 0)
            stream << std::endl;
    }
    stream << std::endl << "};" << std::endl;

    std::string output_content = stream.str();
    std::ofstream file_out(argv[2], std::ofstream::trunc);
    file_out << output_content;

    return EXIT_SUCCESS;
}