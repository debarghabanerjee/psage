#ifndef SHORT_VECTOR_FILE_HPP
#define SHORT_VECTOR_FILE_HPP

#include "Python.h"
#include <vector>
#include <fstream>


class ShortVectorFile 
{
public:
  ShortVectorFile( PyObject*, PyObject*, const uint64_t );
  ShortVectorFile( PyObject* );
  ~ShortVectorFile();
 
  uint64_t  maximal_vector_length() const { return this->maximal_vector_length__cache; };

  PyObject* write_vectors( const uint64_t, PyObject* );
  void increase_maximal_vector_length( const uint64_t );

  template <class T> friend inline ShortVectorFile& operator>>( ShortVectorFile&, T& );
  template <class T> friend inline ShortVectorFile& operator<<( ShortVectorFile&, const T& );

private:
  std::fstream* output_file;
  std::vector<std::vector<int64_t>>* lattice;
  uint64_t maximal_vector_length__cache;

  void read_header();

  void read_lattice( PyObject* );
  void read_lattice();

  void write_header( const uint64_t );

  std::vector<std::pair<size_t, uint64_t>>* data_blocks();
  void move_data_block( size_t, size_t, uint64_t );


  // todo: in order to have a complete and useful class, implement deletion
  // of unused data blocks.
};

#endif
