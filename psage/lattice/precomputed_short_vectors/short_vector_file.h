#ifndef SHORT_VECTOR_FILE_HPP
#define SHORT_VECTOR_FILE_HPP

#include "Python.h"

#include <fstream>
#include <string>
#include <tuple>
#include <vector>

class ShortVectorFile 
{
public:
  ShortVectorFile( const std::string&, const std::vector<std::vector<int>>&, const unsigned int );
  ShortVectorFile( const std::string& );
  ShortVectorFile( PyObject*, PyObject*, const unsigned int );
  ShortVectorFile( PyObject* );
  ~ShortVectorFile();

  void flush() { this->output_file.flush(); };

  void init_with_lattice( const std::string&, const std::vector<std::vector<int>>&, const unsigned int );
  void init_with_file_name( const std::string& );

  const std::vector<std::vector<int>>& get_lattice() const
  {
    return this->lattice;
  };
  PyObject* get_lattice_py() const;
 
  unsigned int maximal_vector_length() const
  {
    return this->maximal_vector_length__cache;
  };
  void increase_maximal_vector_length( const unsigned int );

  PyObject* stored_vectors_py();
  PyObject* write_vectors_py( const unsigned int, PyObject* );
  std::vector<std::vector<int>> read_vectors( const unsigned int );
  PyObject* read_vectors_py( const unsigned int );

  template <class T> friend inline ShortVectorFile& operator>>( ShortVectorFile&, T& );
  template <class T> friend inline ShortVectorFile& operator<<( ShortVectorFile&, const T& );

private:
  std::fstream output_file;
  std::vector<std::vector<int>> lattice;
  // The maximal length that can be stored. The maximum may be attained!
  unsigned int maximal_vector_length__cache;
  std::vector<std::tuple<unsigned int, size_t, size_t>> stored_vectors__cache;
  size_t next_free_position;

  size_t read_header();
  size_t write_header();

  std::vector<std::vector<int>> parse_python_lattice( PyObject* );
  void read_lattice();
  void write_lattice();

  void read_stored_vectors();
  void write_stored_vectors__empty();

  void move_data_block( size_t, size_t, size_t );
};

#endif
