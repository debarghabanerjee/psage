/// Something is completely wrong about get. Find out what exactly.

#include "Python.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "short_vector_file.h"

using namespace std;


ShortVectorFile::ShortVectorFile
(
 const string& output_file_name,
 const vector<vector<int>>& lattice,
 const unsigned int maximal_vector_length
 )
{
  this->init_with_lattice( output_file_name, lattice, maximal_vector_length );
}


ShortVectorFile::ShortVectorFile
(
 const string& output_file_name
 )
{
  this->init_with_file_name( output_file_name );
}

ShortVectorFile::ShortVectorFile(
    PyObject* py_output_file_name,
    PyObject* py_lattice,
    const unsigned int maximal_vector_length
    )
{
    auto lattice = this->parse_python_lattice(py_lattice);

    if ( !PyString_Check( py_output_file_name ) )
    {
        throw( string( "py_output_file_name must be a string" ) );
    }
    string output_file_name = string( PyString_AsString( py_output_file_name ) );

    this->init_with_lattice( output_file_name, lattice, maximal_vector_length );
}

ShortVectorFile::ShortVectorFile(
    PyObject* py_output_file_name
    )
{
    string output_file_name = string( PyString_AsString( py_output_file_name ) );

    this->init_with_file_name( output_file_name );
}

ShortVectorFile::~ShortVectorFile()
{
    this->output_file.close();
}

void
ShortVectorFile::init_with_lattice
(
 const string& output_file_name,
 const vector<vector<int>>& lattice,
 const unsigned int maximal_vector_length
 )
{
  // todo: check whether copy constructor is deep
  for ( auto& row : lattice )
    this->lattice.push_back( vector<int>( row ) );

  this->output_file.open( output_file_name,
			  fstream::in | fstream::out | fstream::binary );
  if (! this->output_file.is_open() )
    throw( string( "output file must exist already" ) );
  
  this->maximal_vector_length__cache = maximal_vector_length;

  this->next_free_position = this->write_header();
}

void
ShortVectorFile::init_with_file_name
(
 const string& output_file_name
 )
{
  this->output_file.open( output_file_name,
			  fstream::in | fstream::out | fstream::app | fstream::binary );
	  
  this->next_free_position = this->read_header();
}

PyObject*
ShortVectorFile::get_lattice_py() const
{
  size_t lattice_rank = this->lattice.size();

  PyObject* lattice_py = PyList_New( lattice_rank );
  PyObject* row_py, *entry_py;

  for ( size_t ind = 0; ind < lattice_rank; ++ind )
    {
      row_py = PyList_New( lattice_rank );
      for ( size_t ind_row = 0; ind_row < lattice_rank; ++ind )
	{
	  entry_py = PyInt_FromLong( this->lattice[ind][ind_row] );
	  PyList_SetItem( row_py, ind_row, entry_py );
	}
    }

  return lattice_py;
}

size_t
ShortVectorFile::read_header()
{
  this->read_lattice();
  uint64_t maximal_vector_length__cache_64;
  *this >> maximal_vector_length__cache_64;
  this->maximal_vector_length__cache = (unsigned int)maximal_vector_length__cache_64;
  this->read_stored_vectors();


  if ( this->stored_vectors__cache.size() == 0 )
    return (1 + this->lattice.size() * this->lattice.size() + 1
	    + 3 * (this->maximal_vector_length__cache << 2) + 1) * sizeof(uint64_t);
  else
    {
      auto last_block = *this->stored_vectors__cache.cend();
      return get<2>( last_block ) + get<1>( last_block ) * this->lattice.size() * sizeof(uint64_t);
    }
}

size_t
ShortVectorFile::write_header()
{
  this->output_file.seekp( 0, ios_base::beg );
  this->write_lattice();
  *this << (uint64_t)this->maximal_vector_length__cache;
  this->write_stored_vectors__empty();

  *this << (uint64_t)0;

  this->output_file.flush();

  return (   1 + this->lattice.size() * this->lattice.size()
           + 1 + 3 * (this->maximal_vector_length__cache / 2) + 1 ) * sizeof(uint64_t);
}

vector<vector<int>>
ShortVectorFile::parse_python_lattice(
    PyObject* py_lattice
    )
{
    Py_ssize_t lattice_rank = PyList_Size( py_lattice );
    vector<vector<int>> lattice;

    long entry;
    PyObject* py_lattice_row;
    PyObject* py_lattice_entry;
    for ( Py_ssize_t i = 0; i < lattice_rank; ++i )
      {
        py_lattice_row = PyList_GetItem( py_lattice, i );
	if ( lattice_rank != PyList_Size( py_lattice_row ) )
	  throw( string( "Rows and columns of lattice must all have the same size." ) );

	vector<int> current_row;
	for ( Py_ssize_t j = 0; j < lattice_rank; ++j )
	  {
	    py_lattice_entry = PyList_GetItem( py_lattice_row, j );
	    if (! PyInt_Check( py_lattice_entry ) )
              throw( string( "Entries of lattice must by Python int's." ) );

	    current_row.push_back( (int)PyInt_AsLong( py_lattice_entry ) );
	  }
	lattice.push_back( current_row );
      }

    return lattice;
}

void
ShortVectorFile::read_lattice()
{
  unsigned int lattice_rank;
  uint64_t lattice_rank_64;

  this->output_file.seekg( 0, ios_base::beg );
  *this >> lattice_rank_64;
  lattice_rank = (unsigned int)lattice_rank_64;

  int64_t entry_64;
  for ( size_t it_row = 0; it_row < lattice_rank; ++it_row )
    {
      vector<int> row;
      this->lattice.push_back( row );
      for ( size_t it_col = 0; it_col < lattice_rank; ++it_col )
        {
	  *this >> entry_64;
	  row.push_back( (int)entry_64 );

	}
    }
}

void
ShortVectorFile::write_lattice()
{
  *this << (uint64_t)this->lattice.size();
  for ( auto row_it : this->lattice )
    for ( auto it : row_it )
      *this << (int64_t)it;
}

void
ShortVectorFile::read_stored_vectors()
{
  uint64_t length_64, nmb_vectors_64, position_64;

  for ( unsigned int it = 0; it < this->maximal_vector_length__cache / 2; ++it )
    {
      *this >> length_64 >> nmb_vectors_64 >> position_64;

      if ( position_64 != 0 )
	this->stored_vectors__cache.push_back
	  ( tuple<unsigned int, size_t, size_t>
	    ( (unsigned int)length_64, (size_t)nmb_vectors_64, (size_t)position_64 ) );
    }

  sort( this->stored_vectors__cache.begin(),
	this->stored_vectors__cache.end(),
	[] (tuple<unsigned int, size_t, size_t> fst,
	    tuple<unsigned int, size_t, size_t> snd)
	{ return get<2>( fst ) < get<2>( snd ); } );
}

void
ShortVectorFile::write_stored_vectors__empty()
{
  for ( unsigned int length = 2;
	length <= this->maximal_vector_length__cache;
	length += 2 )
    *this << (uint64_t)length << (uint64_t)0 << (uint64_t)0;
}

PyObject*
ShortVectorFile::stored_vectors_py()
{
  tuple<unsigned int, size_t, unsigned int> vector_data;
  PyObject* py_vector_data;
  PyObject* py_vector_length;
  PyObject* py_nmb_vectors;
  PyObject* py_stored_vectors = PyList_New( this->stored_vectors__cache.size() );

  for ( size_t it = 0; it < this->stored_vectors__cache.size(); ++it )
    {
      vector_data = this->stored_vectors__cache[it];

      py_vector_length = PyInt_FromLong( get<0>( vector_data ) );
      py_nmb_vectors = PyInt_FromLong( get<1>( vector_data ) );

      py_vector_data = PyTuple_New( 2 );
      PyTuple_SetItem( py_vector_data, 0, py_vector_length );
      PyTuple_SetItem( py_vector_data, 1, py_nmb_vectors );

      PyList_SetItem( py_stored_vectors, it, py_vector_data );
    }

  return py_stored_vectors;
}

vector<vector<int>>
ShortVectorFile::read_vectors
(
 const unsigned int length
 )
{
  size_t nmb_vectors;
  size_t data_position = 0;

  for ( auto& it : this->stored_vectors__cache )
    if ( get<0>( it ) == length )
      {
        nmb_vectors = get<1>( it );
        data_position = get<2>( it );
      }
  if ( data_position == 0 )
    return vector<vector<int>>();

  vector<vector<int>> vectors;
  vector<int> vec;
  int64_t entry_64;
  size_t lattice_rank = this->lattice.size();

  this->output_file.seekg( data_position, ios_base::beg );
  for ( size_t it = 0; it < nmb_vectors; ++it )
    {
      vec = vector<int>();
      for ( size_t it_e = 0; it_e < lattice_rank; ++it_e )
	{
	  *this >> entry_64;
	  vec.push_back( (int)( entry_64 ) );
	}
      vectors.push_back( vec );
    }

  return vectors;
}

PyObject*
ShortVectorFile::read_vectors_py(
    const unsigned int length
    )
{
  size_t nmb_vectors;
  size_t data_position = 0;

  PyObject* py_vectors;
  PyObject* py_vector;
  PyObject* py_entry;
  int64_t entry;

  for ( auto& it : this->stored_vectors__cache )
    if ( get<0>( it ) == length )
      {
        nmb_vectors = get<1>( it );
        data_position = get<2>( it );
      }
  if ( data_position == 0 )
    {
      Py_INCREF( Py_None );
      return Py_None;
    }

  py_vectors = PyList_New( nmb_vectors );
  size_t lattice_rank = this->lattice.size();

  this->output_file.seekg( data_position, ios_base::beg );
  for ( size_t it = 0; it < nmb_vectors; ++it )
    {
      py_vector = PyTuple_New( lattice_rank );
      for ( size_t it_e = 0; it_e < lattice_rank; ++it_e )
        {
          *this >> entry;
          py_entry = PyInt_FromLong( (long)entry );
          PyTuple_SetItem( py_vector, it_e, py_entry );
        }
      PyList_SetItem( py_vectors, it, py_vector );
    }

  return py_vectors;
}

PyObject*
ShortVectorFile::write_vectors_py(
    const unsigned int length,
    PyObject* py_vectors
    )
{
    if (! PyList_Check( py_vectors ) ) 
      {
	Py_INCREF(Py_False);
	return Py_False;
      }
    if ( length > this->maximal_vector_length__cache )
      {
	Py_INCREF(Py_False);
	return Py_False;
      }

    // Verify that this entry has not yet been written.
    for ( auto it : this->stored_vectors__cache )
      if ( get<0>( it ) == length )
      {
	Py_INCREF(Py_False);
	return Py_False;
      }

    // Write the header entry for this set of vectors.
    size_t header_entry_position = (1 + this->lattice.size() * this->lattice.size() + 1 + 3 * (length / 2 - 1) + 1) * sizeof(int64_t);
    
    this->output_file.seekp( header_entry_position, ios_base::beg );
    *this << (uint64_t)PyList_Size( py_vectors );
    this->output_file.seekp( header_entry_position + 8, ios_base::beg );
    *this << (uint64_t)( this->next_free_position );

    PyObject* py_vector;
    PyObject* py_entry;
    int64_t entry_64;

    this->output_file.seekp( this->next_free_position, ios_base::beg );
    try
      {
        for ( Py_ssize_t it = 0; it < PyList_Size( py_vectors ); ++it )
          {
            py_vector = PyList_GetItem( py_vectors, it );
            if (! PyTuple_Check( py_vector ) )
              throw( string( "conversion failed" ) );

            for ( Py_ssize_t it_v = 0; it_v < PyTuple_Size( py_vector ); ++it_v )
              {
                py_entry = PyTuple_GetItem( py_vector, it_v );
                if (! PyInt_Check( py_entry ) )
                  throw( string( "conversion failed" ) );

                entry_64 = (int64_t)PyInt_AsLong( py_entry );
                *this << entry_64;
              }
        }
      }
    catch ( string &e )
      {
        if ( e == string("conversion failed") )
          {
            this->output_file.seekp( header_entry_position, ios_base::beg );
            *this << (uint64_t)0 << (uint64_t)0;

	    Py_INCREF(Py_False);
	    return Py_False;
          }
        else
          throw;
      }
    this->stored_vectors__cache.push_back
      ( tuple<unsigned int, size_t, size_t>
	( length, PyList_Size( py_vectors ), this->next_free_position ) );
    this->next_free_position +=
      PyList_Size( py_vectors ) * this->lattice.size() * sizeof(int64_t);

    this->output_file.flush();

    Py_INCREF(Py_True);
    return Py_True;
}

void
ShortVectorFile::increase_maximal_vector_length(
    const unsigned int maximal_vector_length
    )
{
    if ( this->maximal_vector_length__cache >= maximal_vector_length )
      return;

    size_t additional_size =
      3 * ( (maximal_vector_length - this->maximal_vector_length__cache) / 2 ) * sizeof( int64_t );

    for ( auto& it : this->stored_vectors__cache )
      {
        this->move_data_block( get<2>( it ), get<2>( it ) + additional_size, get<1>( it ) );
        get<2>( it ) += additional_size;
      }

    uint64_t nmb_vectors_64, position_64;
    this->output_file.seekg( ( this->lattice.size() * this->lattice.size() + 2 ) * sizeof(int64_t), ios_base::beg );
    this->output_file.seekp( ( this->lattice.size() * this->lattice.size() + 2 ) * sizeof(int64_t), ios_base::beg );
    for ( unsigned int length = 2; length <= this->maximal_vector_length__cache; length += 2 )
      {
        this->output_file.seekg( sizeof(int64_t), ios_base::cur );
        *this >> nmb_vectors_64 >> position_64;
        if ( position_64 != 0 )
          {
            this->output_file.seekp( -2 * sizeof(int64_t), ios_base::cur );
            *this << nmb_vectors_64 << (uint64_t)(position_64 + additional_size);
          }
      }

    for ( unsigned int length = this->maximal_vector_length__cache + 2;
          length <= maximal_vector_length;
          length += 2 )
      *this << (uint64_t)length << (uint64_t)0 << (uint64_t)0;

    *this << (uint64_t)0;

    this->next_free_position += additional_size;

    this->maximal_vector_length__cache = maximal_vector_length;
    this->output_file.seekp( ( this->lattice.size() * this->lattice.size() + 1 ) * sizeof(int64_t) );
    *this << (uint64_t)maximal_vector_length;

    this->output_file.flush();
}

void
ShortVectorFile::move_data_block(
    size_t src,
    size_t dest,
    size_t nmb_vectors
    )
{
    vector<int64_t> data;
    int64_t value;

    this->output_file.seekg( src, ios_base::beg );
    for ( size_t it = 0; it < nmb_vectors * this->lattice.size(); ++it )
      {
	*this >> value;
	data.push_back(value);
      }

    this->output_file.seekp( dest, ios_base::beg );
    for ( auto it : data )
      *this << it;
}

template <class T>
inline
ShortVectorFile&
operator>>(
    ShortVectorFile& stream,
    T& dest
    )
{
    stream.output_file.read( reinterpret_cast<char*>( &dest ), sizeof(T) );
    return stream;
}

template <class T>
inline
ShortVectorFile&
operator<<(
    ShortVectorFile& stream,
    const T& src
    )
{
    stream.output_file.write( reinterpret_cast<const char*>( &src ), sizeof(T) );
    return stream;
}
