// todo: move folder to precomputeD_short_vectors

#ifndef SHORT_VECTOR_FILE_HPP
#define SHORT_VECTOR_FILE_HPP

#include "Python.h"

#include <fstream>
#include <tuple>
#include <vector>


class ShortVectorFile 
{
public:
  ShortVectorFile( PyObject*, PyObject*, const uint64_t );
  ShortVectorFile( PyObject* );
  // todo: add constructors and then convert cython
  // ShortVectorFile( const vector<vector<int>>, const string, uint64_t );
  // ShortVectorFile( const string );
  ~ShortVectorFile();

  // todo: implement, mind the conversion of int64 to int
  vector<vector<int>> lattice() const;
 
  uint64_t maximal_vector_length() const { return this->maximal_vector_length__cache; };
  void increase_maximal_vector_length( const uint64_t );

  PyObject* stored_vectors();
  PyObject* write_vectors( const uint64_t, PyObject* );
  // todo: rename to read_vectors_py and provide vector<int*> implementation
  PyObject* read_vectors( const uint64_t );

  template <class T> friend inline ShortVectorFile& operator>>( ShortVectorFile&, T& );
  template <class T> friend inline ShortVectorFile& operator<<( ShortVectorFile&, const T& );

private:
  std::fstream output_file;
  std::vector<std::vector<int64_t>> lattice;
  // The maximal length that can be stored. The maximum may be attained!
  uint64_t maximal_vector_length__cache;
  std::vector<std::tuple<uint64_t, size_t, uint64_t>> stored_vectors__cache;
  size_t next_free_position;

  size_t read_header();
  size_t write_header();

  void parse_python_lattice( PyObject* );
  void read_lattice();
  void write_lattice();

  void read_stored_vectors();
  void write_stored_vectors__empty();

  void move_data_block( size_t, size_t, uint64_t );


};

#endif
